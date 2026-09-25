"""OCR-APT's per-node behavioral features

"""

import numpy as np
import torch

from pidsmaker.utils.dataset_utils import get_num_edge_type, get_rel2id
from pidsmaker.utils.utils import get_split_to_files, log, log_tqdm


def _minmax(col):
    lo, hi = col.min(), col.max()
    if hi <= lo:
        return np.zeros_like(col)
    return (col - lo) / (hi - lo)


def main(cfg):
    edge_type_dim = get_num_edge_type(cfg)
    rel2id = get_rel2id(cfg, from_zero=True)
    oc = cfg.featurization.ocrapt_features

    in_counts, out_counts, times = {}, {}, {}

    base_dir = cfg.transformation._graphs_dir
    split_to_files = get_split_to_files(cfg, base_dir)
    all_paths = [p for paths in split_to_files.values() for p in paths]

    for path in log_tqdm(all_paths, desc="Computing OCR-APT behavioral features"):
        g = torch.load(path, weights_only=False)
        for u, v, attr in g.edges(data=True):
            label = attr.get("label")
            if label not in rel2id:
                continue
            idx = rel2id[label]
            t = attr.get("time")

            u, v = int(u), int(v)
            for n in (u, v):
                if n not in in_counts:
                    in_counts[n] = np.zeros(edge_type_dim, dtype=np.float64)
                    out_counts[n] = np.zeros(edge_type_dim, dtype=np.float64)
                    times[n] = []
            out_counts[u][idx] += 1
            in_counts[v][idx] += 1
            if t is not None:
                times[u].append(t)
                times[v].append(t)

    nodes = sorted(in_counts.keys())
    n_nodes = len(nodes)

    hist = np.zeros((n_nodes, 2 * edge_type_dim), dtype=np.float64)
    idle = np.zeros((n_nodes, 3), dtype=np.float64)
    lifespan = np.zeros(n_nodes, dtype=np.float64) if oc.use_lifespan else None
    cum_active = np.zeros(n_nodes, dtype=np.float64) if oc.use_cumulative_active_time else None

    for i, n in enumerate(nodes):
        hist[i, :edge_type_dim] = in_counts[n]
        hist[i, edge_type_dim:] = out_counts[n]
        ts = sorted(times[n])
        if len(ts) >= 2:
            gaps = np.diff(np.asarray(ts, dtype=np.float64))
            idle[i] = [gaps.min(), gaps.max(), gaps.mean()]
            if lifespan is not None:
                lifespan[i] = ts[-1] - ts[0]
            if cum_active is not None:
                # total time between consecutive actions with gaps under one second
                # 1st-percentile-scale are treated as sub-second bursts
                thr = np.quantile(gaps, 0.01) if len(gaps) else 0.0
                cum_active[i] = float(gaps[gaps <= max(thr, 0.0)].sum())

    # L2-normalize each node's histogram
    norms = np.linalg.norm(hist, axis=1, keepdims=True)
    hist = hist / np.where(norms > 0, norms, 1.0)

    # min-max normalize idle stats; column-wise
    for c in range(idle.shape[1]):
        idle[:, c] = _minmax(idle[:, c])
    if lifespan is not None:
        lifespan = _minmax(lifespan)
    if cum_active is not None:
        cum_active = _minmax(cum_active)

    extra_cols = [idle]
    if lifespan is not None:
        extra_cols.append(lifespan.reshape(-1, 1))
    if cum_active is not None:
        extra_cols.append(cum_active.reshape(-1, 1))
    features = np.concatenate([hist] + extra_cols, axis=1)

    indexid2vec = {str(n): features[i] for i, n in enumerate(nodes)}
    log(f"OCR-APT behavioral features: {len(indexid2vec)} nodes, dim={features.shape[1]}")
    return indexid2vec
