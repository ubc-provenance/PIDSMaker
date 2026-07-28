"""Embedding extraction and smart sampling for temporal visualization.

Extracts per-node, per-time-window embeddings from a trained GNN encoder or
raw Word2Vec featurization, applies neighborhood-aware sampling to keep the
dataset manageable, and packages the result for dimensionality reduction.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from pidsmaker.utils.utils import log


@dataclass
class TemporalEmbedding:
    """A single node's embedding snapshot at one time window."""
    node_id: int
    time_window_idx: int
    time_window_label: str
    embedding: np.ndarray
    label: int
    detection_status: int = 0
    anomaly_score: float = 0.0
    top_edge: str = ""
    embedding_hops: list[np.ndarray] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Container for all extracted embeddings + auxiliary data."""
    embeddings: list
    edges: set = field(default_factory=set)


def _edge_type_ids(batch, n):
    """Per-edge 0-based edge-type ids (argmax of the one-hot edge_type), or -1.

    edge_type is [E, num_types] aligned with edge_index columns; -1 marks
    unknown so downstream can render it gracefully.
    """
    et = getattr(batch, "edge_type", None)
    if et is None:
        return np.full(n, -1, dtype=int)
    try:
        if et.ndim == 2:
            return et.argmax(dim=-1).cpu().numpy()
        return et.cpu().numpy().astype(int)
    except Exception:
        return np.full(n, -1, dtype=int)


def extract_encoder_embeddings(
    model,
    test_data,
    device,
    malicious_node_ids: set,
    detected_node_ids: set | None = None,
    node_anomaly_info: dict | None = None,
) -> ExtractionResult:
    """Extract per-node per-time-window embeddings from a trained model."""
    model.eval()
    all_embeddings: list[TemporalEmbedding] = []
    global_edges: set[tuple[int, int, int]] = set()

    tw_idx = 0
    with torch.no_grad():
        for dataset in test_data:
            for batch in dataset:
                batch = batch.to(device)
                h, h_src, h_dst = model.embed(batch, inference=True)

                if hasattr(batch, "original_n_id"):
                    orig_n_id = batch.original_n_id.cpu().numpy()
                else:
                    orig_n_id = np.arange(h.shape[0] if isinstance(h, torch.Tensor) else h[0].shape[0])

                if hasattr(batch, "original_edge_index"):
                    src_g = batch.original_edge_index[0].cpu().numpy()
                    dst_g = batch.original_edge_index[1].cpu().numpy()
                    times = batch.edge_time.cpu().numpy() if hasattr(batch, "edge_time") else np.zeros(len(src_g), dtype=int)
                    ets = _edge_type_ids(batch, len(src_g))
                    for u, v, t, et in zip(src_g, dst_g, times, ets):
                        global_edges.add((int(u), int(v), int(t), int(et)))
                else:
                    src_l = batch.edge_index[0].cpu().numpy()
                    dst_l = batch.edge_index[1].cpu().numpy()
                    times = batch.edge_time.cpu().numpy() if hasattr(batch, "edge_time") else np.zeros(len(src_l), dtype=int)
                    ets = _edge_type_ids(batch, len(src_l))

                    if len(src_l) == 0:
                        pass  # Empty batch — no edges to process
                    elif len(orig_n_id) > 0 and (src_l.max() >= len(orig_n_id) or dst_l.max() >= len(orig_n_id)):
                        # edge_index contains global IDs — use them directly
                        for u, v, t, et in zip(src_l, dst_l, times, ets):
                            global_edges.add((int(u), int(v), int(t), int(et)))
                    else:
                        for u, v, t, et in zip(orig_n_id[src_l], orig_n_id[dst_l], times, ets):
                            global_edges.add((int(u), int(v), int(t), int(et)))

                if isinstance(h, torch.Tensor):
                    h_np = h.cpu().numpy()
                elif isinstance(h, (tuple, list)):
                    h_np = torch.cat([h[0], h[1]], dim=0).cpu().numpy()
                    orig_n_id = np.concatenate([orig_n_id, orig_n_id])
                else:
                    h_np = h.cpu().numpy()

                tw_label = f"TW_{tw_idx:03d}"

                for local_idx in range(len(orig_n_id)):
                    gid = int(orig_n_id[local_idx])
                    gt_label = 1 if gid in malicious_node_ids else 0

                    if detected_node_ids is not None:
                        if gid in malicious_node_ids:
                            det_status = 1 if gid in detected_node_ids else 2
                        else:
                            det_status = 0
                    else:
                        det_status = 0

                    anomaly_score = 0.0
                    top_edge = ""
                    if node_anomaly_info is not None:
                        ninfo = node_anomaly_info.get(gid, {})
                        anomaly_score = ninfo.get("score", 0.0)
                        top_edge = ninfo.get("edge", "")

                    # The encoder exposes only its final output, so each node has a
                    # single embedding "hop".
                    hops = [h_np[local_idx]]

                    all_embeddings.append(
                        TemporalEmbedding(
                            node_id=gid,
                            time_window_idx=tw_idx,
                            time_window_label=tw_label,
                            embedding=h_np[local_idx],
                            label=gt_label,
                            detection_status=det_status,
                            anomaly_score=anomaly_score,
                            top_edge=top_edge,
                            embedding_hops=hops,
                        )
                    )

                batch.to("cpu")
                torch.cuda.empty_cache()
                tw_idx += 1

    log(f"[embed_exporter] Extracted {len(all_embeddings)} temporal embeddings "
        f"across {tw_idx} time windows.  Edges: {len(global_edges)}")
    return ExtractionResult(embeddings=all_embeddings, edges=global_edges)


def _load_edges_from_nx_graphs(cfg) -> "np.ndarray | set":
    """Load edges from preprocessed PyG graphs (faster and consistent with encoder), fallback to NX graphs.
    """
    dataset = cfg.dataset.name
    artifact_dir = getattr(cfg, "_artifact_dir", "/home/artifacts")
    edges: set[tuple[int, int, int]] = set()

    # Strategy 1: Load from PyG preprocessed graphs
    patterns = [
        os.path.join(artifact_dir, "batching", "batching", "*", dataset, "preprocessed_graphs", "viz_test_graphs.pkl"),
        os.path.join(artifact_dir, "batching", "batching", "*", dataset, "preprocessed_graphs", "torch_graphs.pkl")
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            cache_file = matches[0]
            try:
                log(f"[embed_exporter] Extracting edges from preprocessed cache: {cache_file}")
                data = torch.load(cache_file, map_location="cpu")
                test_data = data[0] if len(data) == 2 else data[2]

                # Collect one vectorised (E,4) block per batch instead of a Python
                # per-edge `edges.add((int(u),...))` loop. That loop ran once per
                # edge (tens of millions on THEIA/E5) and was the generation
                # bottleneck *and* the reason the console looked frozen — it emits
                # no output for minutes. numpy stack + a single np.unique replaces
                # it, and we log per graph-part so progress is visible.
                edge_arrays = []
                n_parts = len(test_data)
                for pi, dataset_part in enumerate(test_data):
                    for batch in dataset_part:
                        orig_n_id = batch.original_n_id.numpy() if hasattr(batch, "original_n_id") else None
                        if hasattr(batch, "original_edge_index"):
                            src = batch.original_edge_index[0].numpy()
                            dst = batch.original_edge_index[1].numpy()
                        else:
                            src_l = batch.edge_index[0].numpy()
                            dst_l = batch.edge_index[1].numpy()
                            if len(src_l) == 0:
                                continue
                            # map local batch ids to global node ids when a valid
                            # mapping exists; otherwise the ids are already global.
                            if (orig_n_id is not None and len(orig_n_id) > 0
                                    and src_l.max() < len(orig_n_id) and dst_l.max() < len(orig_n_id)):
                                src = orig_n_id[src_l]
                                dst = orig_n_id[dst_l]
                            elif orig_n_id is not None:
                                src, dst = src_l, dst_l
                            else:
                                continue
                        if len(src) == 0:
                            continue
                        times = batch.edge_time.numpy() if hasattr(batch, "edge_time") else np.zeros(len(src), dtype=np.int64)
                        ets = np.asarray(_edge_type_ids(batch, len(src)))
                        edge_arrays.append(np.stack([src, dst, times, ets], axis=1).astype(np.int64, copy=False))
                    raw = sum(len(a) for a in edge_arrays)
                    log(f"[embed_exporter] edges: scanned graph part {pi + 1}/{n_parts} ({raw:,} raw edges so far)")
                if edge_arrays:
                    stacked = np.concatenate(edge_arrays)
                    log(f"[embed_exporter] Deduplicating {len(stacked):,} raw edges…")
                    # Dedup with a byte-view 1-D unique (far faster than
                    # np.unique(axis=0) on millions of 4-wide rows) and return the
                    # numpy array directly — building a Python set of millions of
                    # (u,v,t,et) tuples was slow and memory-heavy (looked like a
                    # hang). Consumers only iterate + index + len the result.
                    sv = np.ascontiguousarray(stacked)
                    rows = sv.view(np.dtype((np.void, sv.dtype.itemsize * sv.shape[1]))).reshape(-1)
                    uniq = stacked[np.unique(rows, return_index=True)[1]]
                    log(f"[embed_exporter] Extracted {len(uniq):,} unique edges from cache ({len(stacked):,} raw).")
                    if len(uniq):
                        return uniq
            except Exception as e:
                log(f"[embed_exporter] Error loading PyG cache {cache_file}: {e}")

    # Strategy 2: Fallback to NX graphs
    graphs_dir = cfg.construction._graphs_dir
    # graphs_dir is e.g. /home/artifacts/construction/CADETS_E3/construction/<hash>/nx/
    if not os.path.exists(graphs_dir) or not os.listdir(graphs_dir):
        pattern = os.path.join(artifact_dir, "construction", "*", dataset, "construction", "*", "nx")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(artifact_dir, "construction", dataset, "construction", "*", "nx")
            matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            graphs_dir = matches[0]

    all_paths = [
        p for p in glob.glob(os.path.join(graphs_dir, "**", "*"), recursive=True)
        if os.path.isfile(p) and not p.endswith(".pkl") and not p.endswith(".txt")
    ]
    if not all_paths:
        log(f"[embed_exporter] WARNING: no NX graphs found in {graphs_dir}. ")
        return set()

    for path in all_paths:
        try:
            G = torch.load(path)
            if hasattr(G, "edges"):
                for u, v, data in G.edges(data=True):
                    t = int(data.get("time", 0)) if isinstance(data, dict) else 0
                    edges.add((int(u), int(v), t, -1))  # NX fallback: type unknown
        except Exception as e:
            log(f"[embed_exporter] Error loading {path}: {e}")

    log(f"[embed_exporter] Loaded {len(edges)} edges from {len(all_paths)} NX graph files.")
    return edges


def parse_scores_file(scores_path: str, malicious_node_ids: set) -> tuple[set, set, dict]:
    try:
        data = torch.load(scores_path, map_location="cpu")
        y_preds = data.get("y_preds", [])
        scores = data.get("pred_scores", [])

        involved = set()
        node_anomaly_info = {}

        if "edges" in data:
            edges = data["edges"]
            for i in range(len(y_preds)):
                u, v = int(edges[i][0]), int(edges[i][1])
                score = float(scores[i])
                if y_preds[i]:
                    involved.add(u)
                    involved.add(v)

                if u not in node_anomaly_info or score > node_anomaly_info[u]["score"]:
                    node_anomaly_info[u] = {"score": score, "edge": f"{u} -> {v}"}
                if v not in node_anomaly_info or score > node_anomaly_info[v]["score"]:
                    node_anomaly_info[v] = {"score": score, "edge": f"{u} -> {v}"}
        elif "nodes" in data:
            nodes = data["nodes"]
            for i in range(len(y_preds)):
                u = int(nodes[i])
                score = float(scores[i])
                if y_preds[i]:
                    involved.add(u)

                if u not in node_anomaly_info or score > node_anomaly_info[u]["score"]:
                    node_anomaly_info[u] = {"score": score, "edge": f"Node {u}"}

        detected   = malicious_node_ids & involved
        undetected = malicious_node_ids - involved

        log(
            f"[embed_exporter] Detection split: "
            f"{len(detected)} detected, {len(undetected)} undetected"
        )
        return detected, undetected, node_anomaly_info
    except Exception as e:
        log(f"[embed_exporter] Detection split failed ({e}) — treating all as detected.")
        return malicious_node_ids, set(), {}


# ── Word2Vec extractor ────────────────────────────────────────────────────────

def extract_word2vec_embeddings(
    cfg,
    malicious_node_ids: set,
) -> ExtractionResult:
    """Extract raw Word2Vec node embeddings (static — single time window).

    **Speed fix**: edges are now loaded from the lightweight NX construction
    graph pickles. Word2Vec embeddings are also cached to disk to avoid spending
    2+ minutes tokenizing strings with NLTK on every run.
    """
    import pickle

    # Try to load cached indexid2vec first
    cache_dir = getattr(cfg, "_artifact_dir", "/home/artifacts")
    cache_path = os.path.join(cache_dir, f"viz_w2v_cache_{cfg.dataset.name}.pkl")

    indexid2vec = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                indexid2vec = pickle.load(fh)
            log(f"[embed_exporter] Loaded {len(indexid2vec)} embeddings from viz cache.")
        except Exception as e:
            log(f"[embed_exporter] Failed to load viz cache: {e}")

    if indexid2vec is None:
        try:
            from pidsmaker.featurization.feat_inference import get_indexid2vec
        except ImportError:
            from pidsmaker.tasks.feat_inference import get_indexid2vec

        indexid2vec = get_indexid2vec(cfg)
        if indexid2vec is None:
            raise ValueError(
                "get_indexid2vec returned None — featurization method may be "
                "'only_type' which doesn't produce per-node embeddings."
            )
        log(f"[embed_exporter] Computed {len(indexid2vec)} raw embeddings via get_indexid2vec()")
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(indexid2vec, fh)
            log(f"[embed_exporter] Saved embeddings to viz cache: {cache_path}")
        except Exception as e:
            log(f"[embed_exporter] Failed to save viz cache: {e}")

    # Word2Vec embeddings are RAW (pre-training): there is no detector here, so
    # nodes are NOT scored. Only label (ground truth) is set; anomaly scores and
    # the detected/undetected split belong to the trained per-epoch encoders.
    all_embeddings: list[TemporalEmbedding] = []
    for node_id, vec in indexid2vec.items():
        node_id = int(node_id)
        if isinstance(vec, torch.Tensor):
            vec = vec.numpy()
        elif not isinstance(vec, np.ndarray):
            vec = np.array(vec)

        gt_label = 1 if node_id in malicious_node_ids else 0

        all_embeddings.append(
            TemporalEmbedding(
                node_id=node_id,
                time_window_idx=0,
                time_window_label="Word2Vec (static)",
                embedding=vec,
                label=gt_label,
                detection_status=0,   # raw embeddings: unscored
                anomaly_score=0.0,
                top_edge="",
                embedding_hops=[vec],
            )
        )

    log(f"[embed_exporter] Extracted {len(all_embeddings)} Word2Vec embeddings.")

    # Fast edge loading — NX pickles, not batched PyG graphs
    global_edges = _load_edges_from_nx_graphs(cfg)

    return ExtractionResult(embeddings=all_embeddings, edges=global_edges)


# ── Smart neighbourhood sampler ───────────────────────────────────────────────

def smart_sample(
    result: ExtractionResult,
    max_benign: int | str,
    max_attack: int | str,
) -> ExtractionResult:
    """Subsample embeddings with neighborhood-aware benign selection."""
    if isinstance(max_benign, str) and max_benign.lower() == "all":
        max_benign = float("inf")
    else:
        max_benign = int(max_benign)

    if isinstance(max_attack, str) and max_attack.lower() == "all":
        max_attack = float("inf")
    else:
        max_attack = int(max_attack)

    attack_embs = [e for e in result.embeddings if e.label == 1]
    benign_embs = [e for e in result.embeddings if e.label == 0]

    attack_node_ids = list({e.node_id for e in attack_embs})
    if len(attack_node_ids) > max_attack:
        np.random.seed(42)
        attack_node_ids = list(np.random.choice(attack_node_ids, int(max_attack), replace=False))
        attack_keep = set(attack_node_ids)
        attack_embs = [e for e in attack_embs if e.node_id in attack_keep]

    attack_id_set = {e.node_id for e in attack_embs}

    attack_neighbors: set[int] = set()
    for edge in result.edges:
        u, v = edge[0], edge[1]
        if u in attack_id_set:
            attack_neighbors.add(v)
        if v in attack_id_set:
            attack_neighbors.add(u)
    attack_neighbors -= attack_id_set

    benign_node_ids = list({e.node_id for e in benign_embs})

    if len(benign_node_ids) > max_benign:
        np.random.seed(42)
        neighbor_ids = list(attack_neighbors & set(benign_node_ids))
        if len(neighbor_ids) > max_benign:
            neighbor_ids = list(np.random.choice(neighbor_ids, int(max_benign), replace=False))

        remaining_budget = int(max_benign) - len(neighbor_ids)
        remaining_ids = [n for n in benign_node_ids if n not in set(neighbor_ids)]

        if remaining_budget > 0 and remaining_ids:
            extra = list(np.random.choice(
                remaining_ids,
                min(remaining_budget, len(remaining_ids)),
                replace=False,
            ))
            neighbor_ids.extend(extra)

        benign_keep = set(neighbor_ids)
        benign_embs = [e for e in benign_embs if e.node_id in benign_keep]

    sampled = attack_embs + benign_embs
    log(f"[embed_exporter] Sampled: {len(attack_embs)} attack entries, "
        f"{len(benign_embs)} benign entries "
        f"({len({e.node_id for e in attack_embs})} attack nodes, "
        f"{len({e.node_id for e in benign_embs})} benign nodes)")

    return ExtractionResult(embeddings=sampled, edges=result.edges)
