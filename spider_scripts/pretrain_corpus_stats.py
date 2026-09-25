"""Compute comprehensive statistics over the pretrain corpus datasets.

Directly discovers most-recently-completed build_graphs artifacts on disk,
bypassing the config hash system.

Usage:
    cd /home/pids
    python scripts/pretrain_corpus_stats.py
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_ROOT = "/home/artifacts/preprocessing"
PRETRAIN_DATASETS = [
    "PROVENANCE_BENIGN",
    "TRACE_E3",
    "TRACE_E5",
    "CADETS_E5",
    "CLEARSCOPE_E3",
    "optc_h201",
]

# Dataset split configs (from pidsmaker/config/config.py)
DATASET_SPLITS = {
    "PROVENANCE_BENIGN": {
        "train_files": ["graph_16"],
        "val_files":   ["graph_16"],
        "test_files":  ["graph_16"],
    },
    "TRACE_E3": {
        "train_files": ["graph_2","graph_3","graph_4","graph_5","graph_6","graph_7","graph_8"],
        "val_files":   ["graph_9"],
        "test_files":  ["graph_10","graph_11","graph_12","graph_13"],
    },
    "TRACE_E5": {
        "train_files": ["graph_8","graph_9","graph_10","graph_11","graph_12"],
        "val_files":   ["graph_13"],
        "test_files":  ["graph_14","graph_15"],
    },
    "CADETS_E5": {
        "train_files": ["graph_8","graph_9","graph_11"],
        "val_files":   ["graph_12"],
        "test_files":  ["graph_16","graph_17"],
    },
    "CLEARSCOPE_E3": {
        "train_files": ["graph_3","graph_4","graph_5","graph_7","graph_8","graph_9","graph_10"],
        "val_files":   ["graph_2"],
        "test_files":  ["graph_11","graph_12"],
    },
    "optc_h201": {
        "train_files": ["graph_19","graph_20","graph_21"],
        "val_files":   ["graph_22"],
        "test_files":  ["graph_23","graph_24","graph_25"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_build_dir(ds_name):
    """Return the most recently completed build_graphs directory that has graph files."""
    base = os.path.join(ARTIFACTS_ROOT, ds_name, "build_graphs")
    candidates = []
    for h in os.listdir(base):
        path = os.path.join(base, h)
        done = os.path.join(path, "done.txt")
        nx_dir = os.path.join(path, "nx")
        if os.path.exists(done):
            has_graphs = os.path.isdir(nx_dir) and any(
                os.path.isdir(os.path.join(nx_dir, d)) for d in os.listdir(nx_dir)
            ) if os.path.isdir(nx_dir) else False
            candidates.append((os.path.getmtime(done), has_graphs, path))
    if not candidates:
        return None
    # Prefer: has_graphs=True, then most recent
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return candidates[0][2]


def list_graph_files(nx_dir, day_folders):
    """Return all graph-window files under the given day sub-folders."""
    paths = []
    for day in day_folders:
        day_dir = os.path.join(nx_dir, day)
        if os.path.isdir(day_dir):
            for fname in sorted(os.listdir(day_dir)):
                paths.append(os.path.join(day_dir, fname))
    return paths


def extract_date_range(graph_paths):
    """Return (earliest_date, latest_date) strings from file names."""
    all_dates = []
    for p in graph_paths:
        name = os.path.basename(p)
        if "~" in name:
            start, end = name.split("~", 1)
            all_dates.extend([start.strip()[:10], end.strip()[:10]])
    if not all_dates:
        return None, None
    return min(all_dates), max(all_dates)


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(ds_name):
    print(f"\n{'='*70}")
    print(f"  {ds_name}")
    print(f"{'='*70}")

    build_dir = find_latest_build_dir(ds_name)
    if build_dir is None:
        print(f"  ERROR: no completed build_graphs directory found for {ds_name}")
        return None
    print(f"  build_dir: {os.path.basename(build_dir)}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    indexid2msg_path = os.path.join(build_dir, "indexid2msg", "indexid2msg.pkl")
    split2nodes_path = os.path.join(build_dir, "indexid2msg", "split2nodes.pkl")

    print("  Loading indexid2msg...", end="", flush=True)
    indexid2msg = torch.load(indexid2msg_path, map_location="cpu", weights_only=False)
    print(f" {len(indexid2msg):,} entries")

    split2nodes = {}
    if os.path.exists(split2nodes_path):
        split2nodes = torch.load(split2nodes_path, map_location="cpu", weights_only=False)

    splits = DATASET_SPLITS[ds_name]

    # ── Node stats from indexid2msg ───────────────────────────────────────────
    type_counter = Counter()
    label_vocab  = defaultdict(set)
    label_lengths = defaultdict(list)

    for node_id, val in indexid2msg.items():
        ntype, label = val[0], val[1]
        type_counter[ntype] += 1
        label_vocab[ntype].add(label)
        label_lengths[ntype].append(len(label))

    total_nodes = sum(type_counter.values())

    # ── Split node counts ─────────────────────────────────────────────────────
    split_sizes = {s: len(v) for s, v in split2nodes.items()}
    train_node_ids = set()
    for s in split2nodes:
        if any(tf in s or s == "train" for tf in ["train"]):
            train_node_ids |= set(split2nodes[s])
    # Fallback: union of train split keys
    if not train_node_ids and "train" in split2nodes:
        train_node_ids = set(split2nodes["train"])

    # ── Graph files ───────────────────────────────────────────────────────────
    nx_dir = os.path.join(build_dir, "nx")
    train_paths = list_graph_files(nx_dir, splits["train_files"])
    val_paths   = list_graph_files(nx_dir, splits["val_files"])
    test_paths  = list_graph_files(nx_dir, splits["test_files"])

    t0, t1 = extract_date_range(train_paths)

    # ── Edge stats (iterate train graphs) ────────────────────────────────────
    edge_type_ctr = Counter()
    total_edges = 0
    attack_edges = 0
    self_loops = 0
    node_out = defaultdict(int)
    node_in  = defaultdict(int)
    all_timestamps = []

    print(f"  Loading {len(train_paths)} train graph files...", end="", flush=True)
    for i, gpath in enumerate(train_paths):
        if i % 100 == 0 and i > 0:
            print(f" {i}", end="", flush=True)
        try:
            g = torch.load(gpath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"\n    WARN: {gpath}: {e}")
            continue
        for src, dst, _key, edata in g.edges(data=True, keys=True):
            elabel = edata.get("label", "?")
            y      = edata.get("y", 0)
            ts     = edata.get("time", None)
            edge_type_ctr[elabel] += 1
            total_edges += 1
            attack_edges += int(y == 1)
            self_loops   += int(src == dst)
            node_out[src] += 1
            node_in[dst]  += 1
            if ts is not None:
                all_timestamps.append(ts)
    print(" done")

    # Per-type degree distribution
    type_out = defaultdict(list)
    type_in  = defaultdict(list)
    active_nodes = set(node_out) | set(node_in)
    for nid in active_nodes:
        entry = indexid2msg.get(nid) or indexid2msg.get(str(nid))
        ntype = entry[0] if entry else "unknown"
        type_out[ntype].append(node_out.get(nid, 0))
        type_in[ntype].append(node_in.get(nid, 0))

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n  -- Nodes (indexid2msg: {total_nodes:,} total) --")
    for nt in sorted(type_counter):
        pct = 100 * type_counter[nt] / total_nodes
        voc = len(label_vocab[nt])
        avg_l = np.mean(label_lengths[nt])
        max_l = max(label_lengths[nt])
        print(f"    {nt:<10}: {type_counter[nt]:>10,}  ({pct:5.1f}%)  "
              f"vocab={voc:>8,}  avg_len={avg_l:5.1f}  max_len={max_l}")

    print(f"\n  -- Splits (split2nodes) --")
    for s in sorted(split_sizes):
        print(f"    {s:<8}: {split_sizes[s]:>10,} nodes")

    print(f"\n  -- Time windows --")
    print(f"    train: {len(train_paths):>5} files  ({len(splits['train_files'])} day-folders)")
    print(f"    val  : {len(val_paths):>5} files")
    print(f"    test : {len(test_paths):>5} files")
    if t0:
        print(f"    train date range: {t0}  →  {t1}")
    if all_timestamps:
        ts_min_s = min(all_timestamps) / 1e9
        ts_max_s = max(all_timestamps) / 1e9
        duration_h = (ts_max_s - ts_min_s) / 3600
        print(f"    train duration  : {duration_h:.1f} h")

    print(f"\n  -- Edges (train split: {total_edges:,} total) --")
    print(f"    attack (y=1)  : {attack_edges:>12,}  ({100*attack_edges/max(1,total_edges):.4f}%)")
    print(f"    self-loops    : {self_loops:>12,}  ({100*self_loops/max(1,total_edges):.2f}%)")
    print(f"    unique nodes in train graphs: {len(active_nodes):>10,}")
    print(f"    edge types ({len(edge_type_ctr)}):")
    for et, cnt in sorted(edge_type_ctr.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(1, total_edges)
        print(f"      {et:<35} {cnt:>10,}  ({pct:5.2f}%)")

    print(f"\n  -- Degree stats (train graph nodes) --")
    for nt in sorted(type_out):
        out_deg = type_out[nt]
        in_deg  = type_in[nt]
        print(f"    {nt:<10}  out: avg={np.mean(out_deg):6.2f}  med={np.median(out_deg):5.1f}"
              f"  max={max(out_deg):8}  "
              f"|  in: avg={np.mean(in_deg):6.2f}  med={np.median(in_deg):5.1f}"
              f"  max={max(in_deg):8}")

    return {
        "ds_name": ds_name,
        "total_nodes": total_nodes,
        "type_counter": dict(type_counter),
        "label_vocab": {k: len(v) for k, v in label_vocab.items()},
        "label_vocab_sets": label_vocab,
        "split_sizes": split_sizes,
        "train_node_ids": train_node_ids,
        "total_edges": total_edges,
        "attack_edges": attack_edges,
        "edge_type_ctr": dict(edge_type_ctr),
        "n_train_files": len(train_paths),
        "n_val_files": len(val_paths),
        "n_test_files": len(test_paths),
        "date_range": (t0, t1),
        "active_nodes": len(active_nodes),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cross-dataset statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_cross_dataset_stats(all_stats):
    print(f"\n\n{'#'*70}")
    print(f"  COMBINED / CROSS-DATASET STATISTICS")
    print(f"{'#'*70}")

    valid = [s for s in all_stats if s is not None]

    total_nodes = sum(s["total_nodes"] for s in valid)
    total_edges = sum(s["total_edges"] for s in valid)
    total_attack = sum(s["attack_edges"] for s in valid)

    # Unified label vocabulary
    combined_vocab = defaultdict(set)
    for s in valid:
        for nt, vset in s["label_vocab_sets"].items():
            combined_vocab[nt] |= vset

    # Aggregate edge type counts
    agg_edge = Counter()
    for s in valid:
        for et, cnt in s["edge_type_ctr"].items():
            agg_edge[et] += cnt

    print(f"\n  -- Global totals --")
    print(f"    Datasets processed      : {len(valid)}")
    print(f"    Total indexid2msg nodes : {total_nodes:>12,}")
    print(f"    Total train edges       : {total_edges:>12,}")
    print(f"    Total attack edges      : {total_attack:>12,}  ({100*total_attack/max(1,total_edges):.4f}%)")

    print(f"\n  -- Combined label vocabulary --")
    for nt in sorted(combined_vocab):
        print(f"    {nt:<10}: {len(combined_vocab[nt]):>10,} unique labels")

    print(f"\n  -- Combined edge type distribution --")
    for et, cnt in sorted(agg_edge.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(1, total_edges)
        print(f"    {et:<35} {cnt:>12,}  ({pct:5.2f}%)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  -- Per-dataset summary table --")
    hdr = (f"  {'Dataset':<22} {'Nodes':>9} {'TrainEdge':>11} {'Attack%':>8} "
           f"{'subj-voc':>10} {'file-voc':>10} {'net-voc':>8} "
           f"{'TrainFiles':>11} {'DateRange'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for s in valid:
        vc = s["label_vocab"]
        pct = 100 * s["attack_edges"] / max(1, s["total_edges"])
        dr = f"{s['date_range'][0]} → {s['date_range'][1]}" if s["date_range"][0] else "N/A"
        print(f"  {s['ds_name']:<22} "
              f"{s['total_nodes']:>9,} "
              f"{s['total_edges']:>11,} "
              f"{pct:>7.4f}% "
              f"{vc.get('subject', 0):>10,} "
              f"{vc.get('file', 0):>10,} "
              f"{vc.get('netflow', 0):>8,} "
              f"{s['n_train_files']:>11} "
              f"{dr}")

    # ── Pairwise label Jaccard ────────────────────────────────────────────────
    print(f"\n  -- Pairwise label Jaccard similarity per node type --")
    for nt in sorted(combined_vocab):
        print(f"\n    node_type = {nt}")
        for i, si in enumerate(valid):
            vi = si["label_vocab_sets"].get(nt, set())
            for j, sj in enumerate(valid):
                if j <= i:
                    continue
                vj = sj["label_vocab_sets"].get(nt, set())
                union = vi | vj
                inter = vi & vj
                jac = len(inter) / len(union) if union else 0.0
                print(f"      {si['ds_name']:<22} ∩ {sj['ds_name']:<22}: "
                      f"J={jac:.4f}  shared={len(inter):,}  "
                      f"|A|={len(vi):,}  |B|={len(vj):,}")

    # ── Walk corpus size estimate ─────────────────────────────────────────────
    # Assume walk params from config: num_walks=3, walk_length=5
    num_walks = 3
    walk_length = 5
    total_active = sum(s["active_nodes"] for s in valid)
    estimated_raw_walks = total_active * num_walks * 2   # forward + backward
    print(f"\n  -- Walk corpus size estimate (num_walks={num_walks}, walk_len={walk_length}) --")
    print(f"    Unique active nodes (train, across datasets): {total_active:>10,}")
    print(f"    Raw walks before dedup (×2 directions)      : {estimated_raw_walks:>10,}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Pretrain datasets: {PRETRAIN_DATASETS}")

    all_stats = []
    for ds_name in PRETRAIN_DATASETS:
        try:
            stats = compute_dataset_stats(ds_name)
            all_stats.append(stats)
        except Exception as exc:
            import traceback
            print(f"\nERROR processing {ds_name}:")
            traceback.print_exc()
            all_stats.append(None)

    compute_cross_dataset_stats(all_stats)
    print(f"\nDone.")
