"""Compute OS-level node/edge/unique counts for the SPIDER pretrain corpus.

Unique nodes = unique (node_type, label) tuples per OS (intra-OS dedup across
datasets). This matches the ProvenanceTokenizer's deduplication policy.
"""

import os
from collections import defaultdict

import torch

ARTIFACTS_ROOT = "/home/artifacts/preprocessing"

OS_OF = {
    "PROVENANCE_BENIGN": "Linux",
    "TRACE_E3":          "Linux",
    "TRACE_E5":          "Linux",
    "CADETS_E5":         "FreeBSD",
    "CLEARSCOPE_E3":     "Android",
    "optc_h201":         "Windows",
}

# Pre-computed train-edge counts (from pretrain_corpus_stats.py run).
TRAIN_EDGES = {
    "PROVENANCE_BENIGN":      166_662,
    "TRACE_E3":            10_867_323,
    "TRACE_E5":           231_338_136,
    "CADETS_E5":           37_900_710,
    "CLEARSCOPE_E3":          499_738,
    "optc_h201":            6_493_345,
}


def find_build_dir(ds_name):
    base = os.path.join(ARTIFACTS_ROOT, ds_name, "build_graphs")
    cands = []
    for h in os.listdir(base):
        path = os.path.join(base, h)
        done = os.path.join(path, "done.txt")
        idx  = os.path.join(path, "indexid2msg", "indexid2msg.pkl")
        if os.path.exists(done) and os.path.exists(idx):
            cands.append((os.path.getmtime(done), path))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


# ── Load + group by OS ─────────────────────────────────────────────────────
per_ds_nodes = {}
per_os_unique_labels = defaultdict(set)        # OS → set of (node_type, label)
per_os_node_count = defaultdict(int)
per_os_edge_count = defaultdict(int)

for ds_name, os_name in OS_OF.items():
    print(f"Loading {ds_name}...", end="", flush=True)
    bd = find_build_dir(ds_name)
    idx_path = os.path.join(bd, "indexid2msg", "indexid2msg.pkl")
    indexid2msg = torch.load(idx_path, map_location="cpu", weights_only=False)
    print(f" {len(indexid2msg):,} nodes")

    per_ds_nodes[ds_name] = len(indexid2msg)
    per_os_node_count[os_name] += len(indexid2msg)
    per_os_edge_count[os_name] += TRAIN_EDGES[ds_name]
    for _nid, val in indexid2msg.items():
        per_os_unique_labels[os_name].add((val[0], val[1]))

# ── Totals ─────────────────────────────────────────────────────────────────
total_nodes = sum(per_os_node_count.values())
total_edges = sum(per_os_edge_count.values())
total_unique = sum(len(s) for s in per_os_unique_labels.values())   # NB: no cross-OS dedup since OSes have disjoint setups

# ── Print table ────────────────────────────────────────────────────────────
print()
print(f"{'OS':<10} {'Nodes':>14} {'%nodes':>8} {'UniqLabels':>14} {'%uniq':>7} "
      f"{'Edges':>14} {'%edges':>8}")
print("-" * 80)
for os_name in ["Linux", "FreeBSD", "Windows", "Android"]:
    n  = per_os_node_count[os_name]
    u  = len(per_os_unique_labels[os_name])
    e  = per_os_edge_count[os_name]
    print(f"{os_name:<10} {n:>14,} {100*n/total_nodes:>7.2f}% "
          f"{u:>14,} {100*u/total_unique:>6.2f}% "
          f"{e:>14,} {100*e/total_edges:>7.2f}%")
print("-" * 80)
print(f"{'Total':<10} {total_nodes:>14,} {100:>7.2f}% "
      f"{total_unique:>14,} {100:>6.2f}% "
      f"{total_edges:>14,} {100:>7.2f}%")
