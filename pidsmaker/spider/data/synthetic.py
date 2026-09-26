"""
Synthetic walk corpus loader for T5 and GNN distillation pretraining augmentation.

Reads *.txt files from a directory where each line is a provenance walk in the format:
    [NODE_TYPE] label | EVENT_TYPE | [NODE_TYPE] label | ...

For T5: extracts FORWARD and BACKWARD directional walk entries.
For GNN distillation: builds per-entity 1-hop adjacency from walk edges, with a
    lightweight SyntheticSampler that provides the same sample_nhop_neighborhood
    interface as ProvenanceWalkSampler.
"""

import glob
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pidsmaker.utils.utils import log


_NODE_PATTERN = re.compile(r"^\[(PROC|FILE|SOCK)\]\s+(.+)$")


def _parse_synthetic_walk(line):
    """Parse a single synthetic walk line.

    Expected format:  [NODE_TYPE] label | EVENT_TYPE | [NODE_TYPE] label | ...
    Nodes are at even positions (0, 2, ...), edges at odd positions (1, 3, ...).

    Returns (node_list, edge_list) where node_list is a list of (node_type, label)
    tuples and edge_list is a list of edge-type strings, or None if malformed.
    """
    parts = [p.strip() for p in line.split(" | ")]
    if len(parts) < 3 or len(parts) % 2 == 0:
        return None

    node_list = []
    edge_list = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            m = _NODE_PATTERN.match(part)
            if not m:
                return None
            node_list.append((m.group(1), m.group(2).strip()))
        else:
            edge_list.append(part)
    return node_list, edge_list


def load_synthetic_t5_corpus(synthetic_dir):
    """Parse all .txt files in synthetic_dir and build a T5-compatible corpus.

    Each non-empty, non-comment line should be a walk:
        [NODE_TYPE] label | EVENT_TYPE | [NODE_TYPE] label | ...

    For each entity at position i in a walk, two directional entries are emitted:
    - FORWARD  (i < last): context_edges = edges[i:],        context_nodes = node_ids[i+1:]
    - BACKWARD (i > 0):    context_edges = reversed(edges[:i]), context_nodes = reversed(node_ids[:i])

    Args:
        synthetic_dir: path to directory containing *.txt walk files.

    Returns:
        (corpus, indexid2msg) where:
            corpus      - list of (entity_id, direction, context_edges, context_nodes,
                          indexid2msg) tuples, compatible with _build_t5_walk_corpus output.
            indexid2msg - dict mapping synthetic node ID -> (node_type, label_str).
    """
    txt_files = sorted(glob.glob(os.path.join(synthetic_dir, "*.txt")))
    if not txt_files:
        log(f"No synthetic .txt files found in {synthetic_dir}")
        return [], {}

    raw_walks = []
    for fpath in txt_files:
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                result = _parse_synthetic_walk(line)
                if result is not None:
                    raw_walks.append(result)

    log(f"Loaded {len(raw_walks):,} synthetic walks from {len(txt_files)} file(s)")

    # Build entity index: (node_type, label) -> unique synthetic ID
    entity_to_id = {}
    indexid2msg = {}

    def _get_id(node_type, label):
        key = (node_type, label)
        if key not in entity_to_id:
            eid = f"synth::{node_type}::{label}"
            entity_to_id[key] = eid
            indexid2msg[eid] = (node_type, label)
        return entity_to_id[key]

    for node_list, _ in raw_walks:
        for node_type, label in node_list:
            _get_id(node_type, label)

    corpus = []
    seen_sigs = set()

    for node_list, edge_list in raw_walks:
        if len(node_list) < 2:
            continue
        node_ids = [_get_id(nt, lb) for nt, lb in node_list]

        for i, entity_id in enumerate(node_ids):
            if i < len(node_ids) - 1:
                ctx_edges = edge_list[i:]
                ctx_nodes = node_ids[i + 1:]
                sig = (entity_id, "forward", tuple(ctx_edges), tuple(ctx_nodes))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    corpus.append((entity_id, "forward", list(ctx_edges), list(ctx_nodes), indexid2msg))

            if i > 0:
                ctx_edges = list(reversed(edge_list[:i]))
                ctx_nodes = list(reversed(node_ids[:i]))
                sig = (entity_id, "backward", tuple(ctx_edges), tuple(ctx_nodes))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    corpus.append((entity_id, "backward", ctx_edges, ctx_nodes, indexid2msg))

    log(f"Synthetic T5 corpus: {len(corpus):,} directional walks for {len(indexid2msg):,} unique entities")
    return corpus, indexid2msg


# ── GNN distillation support ──────────────────────────────────────────────

# Map synthetic [TYPE] to the node_type strings used in real datasets
_SYNTH_TYPE_MAP = {"PROC": "subject", "FILE": "file", "SOCK": "netflow"}


class SyntheticSampler:
    """Lightweight sampler for synthetic walk data.

    Provides the same `sample_nhop_neighborhood` interface as
    ProvenanceWalkSampler, backed by simple adjacency dicts built from
    parsed walks (no timestamps).

    forward_adj[node]  = [(neighbor, edge_type), ...]  outgoing edges
    backward_adj[node] = [(neighbor, edge_type), ...]  incoming edges
    """

    def __init__(self, forward_adj, backward_adj, node_labels):
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj
        self.node_labels = node_labels
        self.nodes = list(set(forward_adj.keys()) | set(backward_adj.keys()))

    def sample_nhop_neighborhood(
        self,
        node: str,
        n_neighbors_min: int = 5,
        n_neighbors_max: int = 20,
        ref_time: Optional[float] = None,
        diverse: bool = False,
    ) -> Optional[Tuple[str, List[str], List[Tuple[str, str, str]]]]:
        """Sample a 1-hop neighborhood from synthetic adjacency.

        Since synthetic data has no timestamps, all neighbors are
        equally eligible. Uses the same diverse selection logic as
        ProvenanceWalkSampler when diverse=True.
        """
        n_neighbors = random.randint(n_neighbors_min, n_neighbors_max)
        half_budget = max(1, n_neighbors // 2)

        bwd_all = self.backward_adj.get(node, [])
        fwd_all = self.forward_adj.get(node, [])

        if not bwd_all and not fwd_all:
            return None

        if diverse:
            bwd_selected = self._diverse_select(bwd_all, half_budget)
            fwd_selected = self._diverse_select(fwd_all, half_budget)
            # Adaptive: if one direction empty, give full budget to other
            if not bwd_selected and fwd_selected and len(fwd_selected) < n_neighbors:
                fwd_selected = self._diverse_select(fwd_all, n_neighbors)
            elif not fwd_selected and bwd_selected and len(bwd_selected) < n_neighbors:
                bwd_selected = self._diverse_select(bwd_all, n_neighbors)
        else:
            bwd_selected = random.sample(bwd_all, min(half_budget, len(bwd_all)))
            fwd_selected = random.sample(fwd_all, min(half_budget, len(fwd_all)))
            if not bwd_selected and fwd_selected and len(fwd_selected) < n_neighbors:
                fwd_selected = random.sample(fwd_all, min(n_neighbors, len(fwd_all)))
            elif not fwd_selected and bwd_selected and len(bwd_selected) < n_neighbors:
                bwd_selected = random.sample(bwd_all, min(n_neighbors, len(bwd_all)))

        all_edges: List[Tuple[str, str, str]] = []
        visited = {node}
        for neighbor, edge_type in bwd_selected:
            all_edges.append((neighbor, node, edge_type))
            visited.add(neighbor)
        for neighbor, edge_type in fwd_selected:
            all_edges.append((node, neighbor, edge_type))
            visited.add(neighbor)

        if len(all_edges) < 1:
            return None

        unique_nodes = [node] + [n for n in visited if n != node]
        return (node, unique_nodes, all_edges)

    def _diverse_select(self, neighbors, budget):
        """Dedup by (edge_type, neighbor_type, label), then round-robin by edge type."""
        if not neighbors:
            return []

        # Dedup
        seen = set()
        deduped = []
        for neighbor, edge_type in neighbors:
            nlbl = self.node_labels.get(neighbor, ("unknown", "unknown"))
            key = (edge_type, nlbl[0], nlbl[1])
            if key not in seen:
                seen.add(key)
                deduped.append((neighbor, edge_type))

        if len(deduped) <= budget:
            return deduped

        # Group by edge type, round-robin
        by_etype: Dict[str, List] = defaultdict(list)
        for item in deduped:
            by_etype[item[1]].append(item)
        for items in by_etype.values():
            random.shuffle(items)

        selected = []
        etypes = list(by_etype.keys())
        random.shuffle(etypes)
        while len(selected) < budget and etypes:
            next_etypes = []
            for etype in etypes:
                if len(selected) >= budget:
                    break
                if by_etype[etype]:
                    selected.append(by_etype[etype].pop())
                if by_etype[etype]:
                    next_etypes.append(etype)
            etypes = next_etypes

        return selected


def load_synthetic_gnn_corpus(synthetic_dir):
    """Parse synthetic walks and build 1-hop adjacency for GNN distillation.

    For each consecutive pair of entities in a walk, extracts a directed
    edge (src → dst, edge_type). Deduplicates edges per entity so that
    each unique (edge_type, neighbor_type, neighbor_label) appears at most
    once per direction.

    Returns:
        (sampler, indexid2msg) where:
            sampler     - SyntheticSampler with sample_nhop_neighborhood interface.
            indexid2msg - dict mapping synthetic node ID → (node_type, label_str).
    """
    txt_files = sorted(glob.glob(os.path.join(synthetic_dir, "*.txt")))
    if not txt_files:
        log(f"No synthetic .txt files found in {synthetic_dir}")
        return None, {}

    raw_walks = []
    for fpath in txt_files:
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                result = _parse_synthetic_walk(line)
                if result is not None:
                    raw_walks.append(result)

    log(f"Loaded {len(raw_walks):,} synthetic walks from {len(txt_files)} file(s)")

    # Build entity index: (node_type, label) → unique synthetic ID
    entity_to_id = {}
    indexid2msg = {}

    def _get_id(node_type, label):
        mapped_type = _SYNTH_TYPE_MAP.get(node_type, node_type)
        key = (mapped_type, label)
        if key not in entity_to_id:
            eid = f"synth::{mapped_type}::{label}"
            entity_to_id[key] = eid
            indexid2msg[eid] = (mapped_type, label)
        return entity_to_id[key]

    # Build adjacency from walk edges
    # forward_adj[src] = [(dst, edge_type), ...] — deduped
    # backward_adj[dst] = [(src, edge_type), ...] — deduped
    fwd_edges = defaultdict(set)
    bwd_edges = defaultdict(set)

    for node_list, edge_list in raw_walks:
        if len(node_list) < 2:
            continue
        node_ids = [_get_id(nt, lb) for nt, lb in node_list]
        for i in range(len(edge_list)):
            src = node_ids[i]
            dst = node_ids[i + 1]
            etype = edge_list[i]
            fwd_edges[src].add((dst, etype))
            bwd_edges[dst].add((src, etype))

    # Convert sets to lists
    forward_adj = {k: list(v) for k, v in fwd_edges.items()}
    backward_adj = {k: list(v) for k, v in bwd_edges.items()}

    # Node labels for diverse selection
    node_labels = {eid: lbl for eid, lbl in indexid2msg.items()}

    sampler = SyntheticSampler(forward_adj, backward_adj, node_labels)

    n_fwd = sum(len(v) for v in forward_adj.values())
    n_bwd = sum(len(v) for v in backward_adj.values())
    log(f"Synthetic GNN corpus: {len(indexid2msg):,} entities, "
        f"{n_fwd:,} fwd edges, {n_bwd:,} bwd edges (deduped)")

    return sampler, indexid2msg
