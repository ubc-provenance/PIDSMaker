"""Dimensionality reduction for temporal 3D embedding visualization.

Supports GPU-accelerated UMAP via:
  1. RAPIDS cuML (if installed) — full GPU UMAP
  2. PyTorch GPU kNN + CPU UMAP optimization — hybrid approach
  3. Pure CPU fallback
"""

import time

import numpy as np

from pidsmaker.utils.utils import log
from pidsmaker.vizgen.embed_exporter import ExtractionResult

# Globals for interactive Viz Studio overrides
GLOBAL_N_NEIGHBORS = None
GLOBAL_MIN_DIST = None
PROGRESS_CALLBACK = None

def _report_progress(msg):
    if PROGRESS_CALLBACK:
        PROGRESS_CALLBACK(msg)
    log(msg)


def reduce_to_3d(result, method="umap", device=None):
    """Project embeddings to 3D via dimensionality reduction.

    Deduplicates identical embedding vectors before running UMAP for massive
    speedup and cleaner cluster structure (provenance datasets often have
    hundreds of thousands of nodes sharing the same Word2Vec embedding).

    Args:
        result: ExtractionResult with embeddings
        method: 'umap' or 'tsne'
        device: torch device string (e.g. 'cuda:0') for GPU acceleration
    """
    embeddings = result.embeddings
    if not embeddings:
        raise ValueError("No embeddings to reduce.")

    num_hops = len(embeddings[0].embedding_hops) if hasattr(embeddings[0], 'embedding_hops') and embeddings[0].embedding_hops else 1
    coords_3d_hops = []

    for hop_idx in range(num_hops):
        if num_hops > 1:
            log(f"--- Running UMAP for Hop {hop_idx}/{num_hops-1} ---")

        if hasattr(embeddings[0], 'embedding_hops') and embeddings[0].embedding_hops:
            X = np.stack([e.embedding_hops[hop_idx] for e in embeddings], axis=0).astype(np.float32)
        else:
            X = np.stack([e.embedding for e in embeddings], axis=0).astype(np.float32)

        n_samples, n_features = X.shape
        if hop_idx == 0:
            log(f"[dim_reduction] Reducing {n_samples} embeddings ({n_features}D) via {method}...")

        # Dedup identical (to 6 dp) vectors so UMAP only reduces the unique set.
        # View each row as one opaque bytes element and run a 1-D unique — far
        # faster than np.unique(axis=0)'s per-element lexicographic sort on wide
        # (128-D) vectors (~11x on 643k rows), with the same result.
        X_rounded = np.ascontiguousarray(np.round(X, decimals=6))
        row_view = X_rounded.view(
            np.dtype((np.void, X_rounded.dtype.itemsize * X_rounded.shape[1]))
        ).reshape(-1)
        _, unique_idx, inverse_idx = np.unique(
            row_view, return_index=True, return_inverse=True
        )
        X_unique = X[unique_idx]
        n_unique = len(X_unique)

        if hop_idx == 0:
            log(f"[dim_reduction] Deduplicated: {n_samples} -> {n_unique} unique vectors")

        std = X_unique.std(axis=0)
        constant_mask = std < 1e-8
        if constant_mask.all():
            log("[dim_reduction] WARNING: All features are constant. Using random projection.")
            np.random.seed(42 + hop_idx)
            coords_unique = np.random.randn(n_unique, 3).astype(np.float32) * 0.01
        elif constant_mask.any():
            coords_unique = _run_reduction(X_unique[:, ~constant_mask], method, n_unique, device)
        else:
            coords_unique = _run_reduction(X_unique, method, n_unique, device)

        coords_3d = coords_unique[inverse_idx]

        np.random.seed(42 + hop_idx)
        jitter_scale = np.std(coords_unique, axis=0) * 0.02
        jitter = np.random.randn(n_samples, 3).astype(np.float32) * jitter_scale
        coords_3d = coords_3d + jitter
        coords_3d -= coords_3d.mean(axis=0)
        coords_3d_hops.append(coords_3d)

    points = []
    for i, emb in enumerate(embeddings):
        points.append({
            "node_id": int(emb.node_id),
            "coords_hops": [[float(coords_3d_hops[h][i, 0]), float(coords_3d_hops[h][i, 1]), float(coords_3d_hops[h][i, 2])] for h in range(num_hops)],
            "tw_idx": int(emb.time_window_idx),
            "tw_label": emb.time_window_label,
            "label": int(emb.label),
            "detection_status": int(emb.detection_status),
            "anomaly_score": float(getattr(emb, 'anomaly_score', 0.0)),
            "top_edge": getattr(emb, 'top_edge', ""),
        })

    log(f"[dim_reduction] Reduced to {len(points)} 3D points across {num_hops} hops.")
    return points


def _gpu_knn(X, n_neighbors, device_str):
    """Compute exact k-nearest neighbors on GPU using PyTorch batched cdist.

    For 640K × 128D, the full distance matrix would be ~1.5TB.
    We batch it to fit in GPU memory (~24GB on RTX 3090).
    """
    import torch

    device = torch.device(device_str)
    X_gpu = torch.from_numpy(X).to(device)
    n = X_gpu.shape[0]
    k = n_neighbors

    # Determine batch size based on available GPU memory
    free_mem = torch.cuda.get_device_properties(device).total_memory
    # Each row of distance matrix: n * 4 bytes (float32)
    # We want batch_size * n * 4 < 20% of GPU mem to be safe, capped at 1000
    batch_size = max(1, int(0.2 * free_mem / (n * 4)))
    batch_size = min(batch_size, 1000, n)

    _report_progress(f"GPU kNN: {n} points, k={k}, batch_size={batch_size}")

    knn_indices = np.empty((n, k), dtype=np.int64)
    knn_dists = np.empty((n, k), dtype=np.float32)

    from tqdm import tqdm

    t0 = time.time()
    for start in tqdm(range(0, n, batch_size), desc="GPU kNN"):
        end = min(start + batch_size, n)
        # Compute pairwise distances for this batch
        dists = torch.cdist(X_gpu[start:end], X_gpu)  # (batch, n)

        # Vectorized self-distance masking (avoids launching thousands of micro-kernels)
        idx = torch.arange(end - start, device=dists.device)
        dists[idx, start + idx] = float('inf')

        # Get top-k nearest
        topk_dists, topk_idx = dists.topk(k, largest=False)
        knn_indices[start:end] = topk_idx.cpu().numpy()
        knn_dists[start:end] = topk_dists.cpu().numpy()

    elapsed = time.time() - t0
    _report_progress(f"GPU kNN completed in {elapsed:.1f}s")

    del X_gpu
    torch.cuda.empty_cache()

    return knn_indices, knn_dists


def _run_reduction(X, method, n_samples, device=None):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    if method == "umap":
        n_neighbors = GLOBAL_N_NEIGHBORS if GLOBAL_N_NEIGHBORS else min(30, n_samples - 1)
        min_dist = GLOBAL_MIN_DIST if GLOBAL_MIN_DIST else 0.5

        # Strategy 1: Try RAPIDS cuML (full GPU UMAP)
        try:
            from cuml.manifold import UMAP as cuUMAP
            _report_progress("Using RAPIDS cuML GPU UMAP")
            t0 = time.time()
            reducer = cuUMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                spread=3.0, random_state=42, init='spectral',
            )
            result = reducer.fit_transform(X)
            _report_progress(f"cuML GPU UMAP completed in {time.time()-t0:.1f}s")
            return np.asarray(result)
        except ImportError:
            pass
        except Exception as e:
            # cuML installed but unusable at runtime; fall back instead of crashing.
            _report_progress(f"RAPIDS cuML GPU UMAP unavailable ({type(e).__name__}: {e}); "
                              f"falling back to CPU-capable UMAP")

        # Strategy 2: GPU kNN precomputation + CPU UMAP optimization
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")



        use_gpu_knn = False
        if device and 'cuda' in str(device):
            try:
                import torch
                if torch.cuda.is_available():
                    use_gpu_knn = True
            except ImportError:
                pass

        if use_gpu_knn and n_samples > 10000:
            _report_progress(f"Step 1/2: GPU kNN search ({n_samples} points)...")
            t0 = time.time()

            knn_indices, knn_dists = _gpu_knn(X, n_neighbors, str(device))

            _report_progress(f"Step 2/2: CPU UMAP optimization (kNN precomputed)...")
            # Feed precomputed kNN into UMAP — skips the expensive NN search
            reducer = umap.UMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                metric="euclidean", random_state=42,
                precomputed_knn=(knn_indices, knn_dists, None),
                n_epochs=200,  # Reduce from 500 default for large data
                verbose=True,  # Enabled tqdm progress bar
            )
            result = reducer.fit_transform(X)
            _report_progress(f"GPU-hybrid UMAP completed in {time.time()-t0:.1f}s total")
            return result
        else:
            # Strategy 3: Pure CPU fallback
            if n_samples > 100000:
                _report_progress(f"WARNING: Running UMAP on {n_samples} points on CPU. This may take hours.")
            t0 = time.time()
            reducer = umap.UMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                metric="euclidean", random_state=42,
                verbose=True,
            )
            result = reducer.fit_transform(X)
            _report_progress(f"CPU UMAP completed in {time.time()-t0:.1f}s")
            return result

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, n_samples // 4))
        reducer = TSNE(
            n_components=3, perplexity=perplexity, random_state=42,
            init="pca" if X.shape[1] >= 3 else "random", learning_rate="auto",
        )
        return reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown reduction method: {method!r}. Use 'umap' or 'tsne'.")
