"""Shared helpers for SPIDER pretraining modules.

Contains walk corpus construction, multiprocessing workers, graph merging,
corpus dumping, KNN evaluation, and dataset loading utilities.
"""

import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import torch

from pidsmaker.utils.utils import get_all_files_from_folders, get_indexid2msg, get_split2nodes, log
from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.config.pipeline import set_dataset_cfg, set_task_paths

from ..data.edge_filter import filter_edges, filter_walk, is_noisy_edge
from ..data.sampler import ProvenanceWalkSampler
from ..training_utils import pad_token_id_lists


# Module-level globals for fork-based multiprocessing (read-only COW).
_corpus_sampler = None
_corpus_indexid2msg = None
_corpus_num_walks = 1
_dump_tokenizer = None
_filter_noisy_edges = False

# Neighborhood sampling globals
_neigh_n_min = 3
_neigh_n_max = 15
_neigh_num_samples = 3
_neigh_shuffle = False


def _prelabel_signature(walk_nodes, walk_edge_types, indexid2msg):
    """Fast pre-filter signature based on node labels (before tokenization).

    This is much faster than tokenization and catches most duplicates early.
    Only walks that pass this filter get tokenized.

    Returns:
        tuple of (node_type, label_str) and edge types
    """
    parts = []
    for i, node_id in enumerate(walk_nodes):
        if node_id in indexid2msg:
            node_type, label_str = indexid2msg[node_id]
            parts.append((node_type, label_str))
        else:
            parts.append(node_id)
        if i < len(walk_edge_types):
            parts.append(walk_edge_types[i])
    return tuple(parts)


def _t5_prelabel_signature(entity_node, direction, context_edges, context_nodes, indexid2msg):
    """Fast pre-filter signature for T5 directional walks."""
    parts = [direction]
    if entity_node in indexid2msg:
        node_type, label_str = indexid2msg[entity_node]
        parts.append((node_type, label_str))
    else:
        parts.append(entity_node)
    for i, nid in enumerate(context_nodes):
        if i < len(context_edges):
            parts.append(context_edges[i])
        if nid in indexid2msg:
            node_type, label_str = indexid2msg[nid]
            parts.append((node_type, label_str))
        else:
            parts.append(nid)
    return tuple(parts)


def _tokenized_walk_signature(walk_nodes, walk_edge_types, indexid2msg, tokenizer):
    """Create a hashable signature for a walk based on TOKENIZED sequence.

    This ensures deduplication happens at the token level (after BPE tokenization),
    not at the node label level. This is critical because different node labels
    can produce identical token sequences after tokenization, leading to repetitive
    training on the same patterns.

    Returns:
        tuple of token IDs representing the tokenized walk
    """
    token_ids, _, _ = tokenizer.tokenize_walk(walk_nodes, walk_edge_types, indexid2msg)
    return tuple(token_ids)


def _corpus_walk_worker(node_chunk):
    """Sample walks for a chunk of nodes (runs in forked child).

    Returns (walks, n_sampled) where walks is a list of
    (walk_nodes, walk_edge_types, prelabel_sig) tuples.
    Each worker does local prelabel dedup to reduce data transfer.
    """
    sampler = _corpus_sampler
    indexid2msg = _corpus_indexid2msg
    num_walks = _corpus_num_walks

    results = []
    local_seen = set()
    n_sampled = 0

    for node in node_chunk:
        for _ in range(num_walks):
            walk_nodes, walk_edge_types, _, entity_pos = sampler._single_walk(node)
            n_sampled += 1
            if len(walk_nodes) <= 1:
                continue
            prelabel_sig = _prelabel_signature(walk_nodes, walk_edge_types, indexid2msg)
            if prelabel_sig not in local_seen:
                local_seen.add(prelabel_sig)
                results.append((walk_nodes, walk_edge_types, prelabel_sig, entity_pos))

    return results, n_sampled


def _t5_corpus_walk_worker(node_chunk):
    """Sample directional walks for a chunk of nodes (T5 mode).

    Returns (walks, n_sampled) where walks is a list of
    (entity_node, direction, context_edges, context_nodes, prelabel_sig) tuples.
    Each worker does local prelabel dedup to reduce data transfer.
    """
    sampler = _corpus_sampler
    indexid2msg = _corpus_indexid2msg
    num_walks = _corpus_num_walks
    do_filter = _filter_noisy_edges

    results = []
    local_seen = set()
    n_sampled = 0

    for node in node_chunk:
        walks = sampler.sample_directional_walks(node, num_walks)
        n_sampled += len(walks)
        for direction, context_edges, context_nodes in walks:
            if do_filter:
                context_edges, context_nodes = filter_walk(
                    context_edges, context_nodes, indexid2msg, entity_node=node,
                )
                if not context_nodes:
                    continue
            sig = _t5_prelabel_signature(node, direction, context_edges, context_nodes, indexid2msg)
            if sig not in local_seen:
                local_seen.add(sig)
                results.append((node, direction, context_edges, context_nodes, sig))

    return results, n_sampled


def _extract_walk_label_corpus(walk_corpus, model_type):
    """Extract (node_type, label_str) pairs from a prelabel-deduped walk corpus.

    Each unique walk contributes one entry per node, giving a distribution that
    reflects actual activity frequency rather than raw entity count.
    """
    label_corpus = []
    if model_type == "t5":
        for entity_node, direction, context_edges, context_nodes, ds_indexid2msg in walk_corpus:
            if entity_node in ds_indexid2msg:
                label_corpus.append(ds_indexid2msg[entity_node])
            for nid in context_nodes:
                if nid in ds_indexid2msg:
                    label_corpus.append(ds_indexid2msg[nid])
    else:
        for walk_nodes, walk_edge_types, ds_indexid2msg, _epos in walk_corpus:
            for node_id in walk_nodes:
                if node_id in ds_indexid2msg:
                    label_corpus.append(ds_indexid2msg[node_id])
    return label_corpus


def _build_comprehensive_walk_corpus(sampler_pairs, tokenizer=None, num_walks=1, walk_length=10,
                                      time_weight="uniform", half_life=1.0, max_retries=50):
    """Build a comprehensive corpus of unique walks ONCE before training.

    Uses fork-based multiprocessing to parallelize walk sampling across CPU
    cores, with fast prelabel-signature deduplication. Each worker samples
    walks for its chunk of nodes with local dedup, then the main process
    does a final global dedup pass.

    Args:
        sampler_pairs: list of (sampler, indexid2msg) tuples for each dataset
        tokenizer: ProvenanceTokenizer for tokenizing walks
        num_walks: target number of unique walks per node
        walk_length: maximum walk length
        time_weight: temporal weighting strategy
        half_life: half-life for exponential weighting
        max_retries: (unused, kept for API compatibility)

    Returns:
        list of (walk_nodes, walk_edge_types, dataset_indexid2msg) tuples
    """
    global _corpus_sampler, _corpus_indexid2msg, _corpus_num_walks

    total_nodes = sum(len(sampler.nodes) for sampler, _ in sampler_pairs)
    log(f"Building walk corpus from {total_nodes:,} nodes...")

    start_time = time.time()
    corpus = []
    seen_prelabel_sigs = set()
    total_generated = 0
    total_unique = 0

    num_workers = min(mp.cpu_count(), 8)

    for sampler, ds_indexid2msg in sampler_pairs:
        all_nodes = sampler.nodes
        _corpus_sampler = sampler
        _corpus_indexid2msg = ds_indexid2msg
        _corpus_num_walks = num_walks

        if num_workers <= 1 or len(all_nodes) < 5000:
            # Sequential fallback for small graphs
            chunk_results = [_corpus_walk_worker(all_nodes)]
        else:
            chunk_size = max(1, (len(all_nodes) + num_workers - 1) // num_workers)
            chunks = [all_nodes[i:i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

            log(f"  Sampling walks with {num_workers} workers across {len(chunks)} chunks...")
            ctx = mp.get_context("fork")
            with ctx.Pool(num_workers) as pool:
                chunk_results = pool.map(_corpus_walk_worker, chunks)

        # Global dedup across chunks
        for walks, n_sampled in chunk_results:
            total_generated += n_sampled
            for walk_nodes, walk_edge_types, prelabel_sig, entity_pos in walks:
                if prelabel_sig not in seen_prelabel_sigs:
                    seen_prelabel_sigs.add(prelabel_sig)
                    corpus.append((walk_nodes, walk_edge_types, ds_indexid2msg, entity_pos))
                    total_unique += 1

    # Clear globals
    _corpus_sampler = None
    _corpus_indexid2msg = None

    elapsed = time.time() - start_time
    dedup_rate = 100 * (1 - total_unique / total_generated) if total_generated > 0 else 0
    log(f"Corpus built: {len(corpus):,} unique walks from {total_generated:,} generated "
        f"({dedup_rate:.1f}% dedup, {elapsed:.1f}s)")
    return corpus


def _build_t5_walk_corpus(sampler_pairs, num_walks):
    """Build a T5-specific corpus of unique directional walks.

    Instead of sampling bidirectional walks and splitting at the entity
    position, directly samples separate forward and backward walks from
    each node.  This produces more diverse training data for the T5
    encoder-decoder architecture.

    Uses fork-based multiprocessing (same pattern as
    _build_comprehensive_walk_corpus).

    Args:
        sampler_pairs: list of (sampler, indexid2msg) tuples per dataset.
        num_walks: walk attempts per direction per node.

    Returns:
        list of (entity_node_id, direction, context_edges, context_nodes,
                 ds_indexid2msg) tuples.
    """
    global _corpus_sampler, _corpus_indexid2msg, _corpus_num_walks

    total_nodes = sum(len(s.nodes) for s, _ in sampler_pairs)
    log(f"Building T5 walk corpus from {total_nodes:,} nodes...")

    start_time = time.time()
    corpus = []
    seen_sigs = set()
    total_generated = 0
    total_unique = 0
    num_workers = min(mp.cpu_count(), 8)

    for sampler, ds_indexid2msg in sampler_pairs:
        all_nodes = sampler.nodes
        _corpus_sampler = sampler
        _corpus_indexid2msg = ds_indexid2msg
        _corpus_num_walks = num_walks

        if num_workers <= 1 or len(all_nodes) < 5000:
            chunk_results = [_t5_corpus_walk_worker(all_nodes)]
        else:
            chunk_size = max(1, (len(all_nodes) + num_workers - 1) // num_workers)
            chunks = [all_nodes[i:i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

            log(f"  Sampling T5 walks with {num_workers} workers across {len(chunks)} chunks...")
            ctx = mp.get_context("fork")
            with ctx.Pool(num_workers) as pool:
                chunk_results = pool.map(_t5_corpus_walk_worker, chunks)

        # Global dedup across chunks
        for walks, n_sampled in chunk_results:
            total_generated += n_sampled
            for node, direction, context_edges, context_nodes, sig in walks:
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    corpus.append((node, direction, context_edges, context_nodes, ds_indexid2msg))
                    total_unique += 1

    # Clear globals
    _corpus_sampler = None
    _corpus_indexid2msg = None

    elapsed = time.time() - start_time
    dedup_rate = 100 * (1 - total_unique / total_generated) if total_generated > 0 else 0
    log(f"T5 corpus: {len(corpus):,} unique directional walks from {total_generated:,} "
        f"({dedup_rate:.1f}% dedup, {elapsed:.1f}s)")
    return corpus


# ── T5 parallel-edge corpus (one training example per edge) ───────────────

def _t5_parallel_edge_worker(node_chunk):
    """Enumerate all unique (entity, direction, edge_type, neighbor) pairs.

    Unlike _t5_neigh_worker which concatenates neighbors into one decoder
    target, this creates one (entity, single_edge) pair per neighbor — removing
    the spurious cross-neighbor autoregressive dependencies.

    Iterates all edges in the adjacency lists directly (no temporal sampling
    or neighborhood size limits), since each edge becomes an independent
    training example.

    Returns (edges, n_total, n_filtered, filtered_examples, skipped_centers)
    where:
      - edges: list of (entity_node, direction_str, [edge_type], [neighbor_node], prelabel_sig)
      - n_total: total edges enumerated (before filtering)
      - n_filtered: edges removed by noise filter
      - filtered_examples: set of (center_label, direction, edge_type, neighbor_label) for summary
      - skipped_centers: set of (center_type, center_label) skipped entirely
    """
    sampler = _corpus_sampler
    indexid2msg = _corpus_indexid2msg
    do_filter = _filter_noisy_edges

    results = []
    local_seen = set()
    n_total = 0
    n_filtered = 0
    filtered_examples = set()
    skipped_centers = set()

    def _label_for(node_id):
        if node_id in indexid2msg:
            ntype, nlabel = indexid2msg[node_id]
            return ntype, nlabel
        return "unknown", str(node_id)

    for node in node_chunk:
        # Skip noisy center entities entirely
        if do_filter and node in indexid2msg:
            ntype, nlabel = indexid2msg[node]
            if ntype == "file":
                from ..data.edge_filter import _is_noisy_file
                if _is_noisy_file(nlabel):
                    skipped_centers.add((ntype, nlabel))
                    continue
            elif ntype == "subject":
                from ..data.edge_filter import _is_noisy_process
                if _is_noisy_process(nlabel):
                    skipped_centers.add((ntype, nlabel))
                    continue

        for forward in (True, False):
            direction = "forward" if forward else "backward"
            adj = sampler.forward_adj if forward else sampler.backward_adj
            if node not in adj:
                continue
            for neigh_node, edge_type, _ts in adj[node]:
                n_total += 1
                if do_filter and is_noisy_edge(node, neigh_node, edge_type, indexid2msg):
                    n_filtered += 1
                    ctype, clabel = _label_for(node)
                    ntype, nlabel = _label_for(neigh_node)
                    filtered_examples.add((
                        f"[{ctype.upper()}] {clabel}",
                        direction,
                        edge_type,
                        f"[{ntype.upper()}] {nlabel}",
                    ))
                    continue
                edge_types = [edge_type]
                neighbor_nodes = [neigh_node]
                sig = _t5_prelabel_signature(
                    node, direction, edge_types, neighbor_nodes, indexid2msg,
                )
                if sig not in local_seen:
                    local_seen.add(sig)
                    results.append((node, direction, edge_types, neighbor_nodes, sig))

    return results, n_total, n_filtered, filtered_examples, skipped_centers


def _build_t5_parallel_edge_corpus(sampler_pairs):
    """Build a T5 corpus with one training example per edge.

    Enumerates all unique (entity, direction, edge_type, neighbor) pairs from
    the adjacency lists. Each edge becomes an independent training example —
    no neighborhood size limits or temporal sampling needed since there are no
    cross-neighbor autoregressive dependencies.

    Uses 'forward'/'backward' direction tokens (same as walks) since each
    example is a single directed hop.

    Returns list of (entity_node, direction, edge_types, neighbor_nodes,
                     ds_indexid2msg) tuples — same schema as walk corpus.
    """
    global _corpus_sampler, _corpus_indexid2msg

    total_nodes = sum(len(s.nodes) for s, _ in sampler_pairs)
    log(f"Building T5 parallel-edge corpus from {total_nodes:,} nodes...")

    start_time = time.time()
    corpus = []
    seen_sigs = set()
    total_edges = 0
    total_filtered = 0
    total_unique = 0
    all_filtered_examples = set()
    all_skipped_centers = set()
    num_workers = min(mp.cpu_count(), 8)

    for sampler, ds_indexid2msg in sampler_pairs:
        all_nodes = sampler.nodes
        _corpus_sampler = sampler
        _corpus_indexid2msg = ds_indexid2msg

        if num_workers <= 1 or len(all_nodes) < 5000:
            chunk_results = [_t5_parallel_edge_worker(all_nodes)]
        else:
            chunk_size = max(1, (len(all_nodes) + num_workers - 1) // num_workers)
            chunks = [all_nodes[i:i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

            log(f"  Sampling parallel edges with {num_workers} workers across {len(chunks)} chunks...")
            ctx = mp.get_context("fork")
            with ctx.Pool(num_workers) as pool:
                chunk_results = pool.map(_t5_parallel_edge_worker, chunks)

        for edges, n_total, n_filtered, filtered_examples, skipped_centers in chunk_results:
            total_edges += n_total
            total_filtered += n_filtered
            all_filtered_examples.update(filtered_examples)
            all_skipped_centers.update(skipped_centers)
            for node, direction, edge_types, neighbor_nodes, sig in edges:
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    corpus.append((node, direction, edge_types, neighbor_nodes, ds_indexid2msg))
                    total_unique += 1

    _corpus_sampler = None
    _corpus_indexid2msg = None

    elapsed = time.time() - start_time
    total_kept = total_edges - total_filtered
    dedup_rate = 100 * (1 - total_unique / total_kept) if total_kept > 0 else 0
    filter_rate = 100 * total_filtered / total_edges if total_edges > 0 else 0
    log(f"T5 parallel-edge corpus: {total_unique:,} unique from {total_edges:,} edges, "
        f"filtered {total_filtered:,} noisy ({filter_rate:.1f}%), "
        f"dedup {dedup_rate:.1f}%, {elapsed:.1f}s")

    # Store filter stats for summary file (written later alongside walk_corpus.txt)
    _build_t5_parallel_edge_corpus.filter_stats = {
        "total_edges": total_edges,
        "filtered": total_filtered,
        "kept": total_kept,
        "unique": total_unique,
        "filter_rate": filter_rate,
        "dedup_rate": dedup_rate,
        "filtered_examples": all_filtered_examples,
        "skipped_centers": all_skipped_centers,
    }
    return corpus


# ── T5 neighborhood corpus (temporal 1-hop neighborhoods) ────────────────

def _t5_neigh_worker(node_chunk):
    """Sample temporal neighborhoods for a chunk of nodes.

    Returns (neighborhoods, n_sampled) where neighborhoods is a list of
    (entity_node, direction_str, edge_types, neighbor_nodes, prelabel_sig).
    """
    sampler = _corpus_sampler
    indexid2msg = _corpus_indexid2msg
    n_min = _neigh_n_min
    n_max = _neigh_n_max
    num_samples = _neigh_num_samples
    shuffle = _neigh_shuffle
    do_filter = _filter_noisy_edges

    results = []
    local_seen = set()
    n_sampled = 0

    for node in node_chunk:
        # Skip noisy center entities entirely
        if do_filter and node in indexid2msg:
            ntype, nlabel = indexid2msg[node]
            if ntype == "file":
                from ..data.edge_filter import _is_noisy_file
                if _is_noisy_file(nlabel):
                    continue
            elif ntype == "subject":
                from ..data.edge_filter import _is_noisy_process
                if _is_noisy_process(nlabel):
                    continue

        for forward in (True, False):
            direction = "forward_neigh" if forward else "backward_neigh"
            for _ in range(num_samples):
                nh = sampler.sample_temporal_neighborhood(
                    node, n_min=n_min, n_max=n_max, forward=forward,
                )
                if nh is None:
                    continue
                n_sampled += 1
                _, _ref_time, neighbors = nh
                if do_filter:
                    neighbors = filter_edges(node, neighbors, indexid2msg)
                    if not neighbors:
                        continue
                if shuffle:
                    neighbors = neighbors.copy()
                    random.shuffle(neighbors)
                edge_types = [e for _, e, _ in neighbors]
                neighbor_nodes = [n for n, _, _ in neighbors]
                sig = _t5_prelabel_signature(
                    node, direction, edge_types, neighbor_nodes, indexid2msg,
                )
                if sig not in local_seen:
                    local_seen.add(sig)
                    results.append((node, direction, edge_types, neighbor_nodes, sig))

    return results, n_sampled


def _build_t5_neighborhood_corpus(sampler_pairs, n_min, n_max, num_samples, shuffle=False):
    """Build a T5-compatible corpus of temporal neighborhood samples.

    For each node, samples both forward and backward 1-hop neighborhoods.
    Returns list of (entity_node, direction, edge_types, neighbor_nodes,
                     ds_indexid2msg) tuples — same schema as walk corpus.
    """
    global _corpus_sampler, _corpus_indexid2msg
    global _neigh_n_min, _neigh_n_max, _neigh_num_samples, _neigh_shuffle

    _neigh_n_min = n_min
    _neigh_n_max = n_max
    _neigh_num_samples = num_samples
    _neigh_shuffle = shuffle

    total_nodes = sum(len(s.nodes) for s, _ in sampler_pairs)
    log(f"Building T5 neighborhood corpus from {total_nodes:,} nodes "
        f"(n_min={n_min}, n_max={n_max}, samples/dir={num_samples})...")

    start_time = time.time()
    corpus = []
    seen_sigs = set()
    total_generated = 0
    total_unique = 0
    num_workers = min(mp.cpu_count(), 8)

    for sampler, ds_indexid2msg in sampler_pairs:
        all_nodes = sampler.nodes
        _corpus_sampler = sampler
        _corpus_indexid2msg = ds_indexid2msg

        if num_workers <= 1 or len(all_nodes) < 5000:
            chunk_results = [_t5_neigh_worker(all_nodes)]
        else:
            chunk_size = max(1, (len(all_nodes) + num_workers - 1) // num_workers)
            chunks = [all_nodes[i:i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

            log(f"  Sampling neighborhoods with {num_workers} workers across {len(chunks)} chunks...")
            ctx = mp.get_context("fork")
            with ctx.Pool(num_workers) as pool:
                chunk_results = pool.map(_t5_neigh_worker, chunks)

        for neighborhoods, n_sampled in chunk_results:
            total_generated += n_sampled
            for node, direction, edge_types, neighbor_nodes, sig in neighborhoods:
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    corpus.append((node, direction, edge_types, neighbor_nodes, ds_indexid2msg))
                    total_unique += 1

    _corpus_sampler = None
    _corpus_indexid2msg = None

    elapsed = time.time() - start_time
    dedup_rate = 100 * (1 - total_unique / total_generated) if total_generated > 0 else 0
    log(f"T5 neighborhood corpus: {len(corpus):,} unique samples from {total_generated:,} "
        f"({dedup_rate:.1f}% dedup, {elapsed:.1f}s)")
    return corpus


def _merge_graphs(graphs):
    """Merge multiple NetworkX MultiDiGraphs into a single graph.

    Node IDs are global (from indexid2msg), so nodes shared across
    time windows are naturally deduplicated. All edges are kept.
    """
    merged = nx.MultiDiGraph()
    for g in graphs:
        for node, attrs in g.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **attrs)
        for src, dst, _, attrs in g.edges(data=True, keys=True):
            merged.add_edge(src, dst, **attrs)
    return merged


def _group_paths_by_day(paths):
    """Group graph file paths by their parent directory (= day folder)."""
    groups = defaultdict(list)
    for p in paths:
        day_folder = os.path.basename(os.path.dirname(p))
        groups[day_folder].append(p)
    return groups


def _load_and_merge_graphs(graph_paths, mode):
    """Load graphs and merge according to graph_context_mode.

    Args:
        graph_paths: list of paths to individual time-window graph files.
        mode: 'window' | 'day' | 'all'.

    Returns:
        List of (potentially merged) graphs.
    """
    if mode == "window":
        return [torch.load(p) for p in graph_paths]

    if mode == "all":
        all_graphs = [torch.load(p) for p in graph_paths]
        merged = _merge_graphs(all_graphs)
        log(f"Merged {len(all_graphs)} graphs → {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
        return [merged]

    # mode == "day"
    day_groups = _group_paths_by_day(graph_paths)
    merged_graphs = []
    for day_folder in sorted(day_groups.keys()):
        day_paths = day_groups[day_folder]
        day_graphs = [torch.load(p) for p in day_paths]
        merged = _merge_graphs(day_graphs)
        merged_graphs.append(merged)
    log(f"Merged {len(day_groups)} days from {len(graph_paths)} windows")
    return merged_graphs


def _dump_walk_worker(args, tokenizer):
    walk_nodes, walk_edge_types, ds_indexid2msg = args[0], args[1], args[2]

    raw_parts = []
    for i, node_id in enumerate(walk_nodes):
        if node_id in ds_indexid2msg:
            node_type, label_str = ds_indexid2msg[node_id]
            raw_parts.append(f"[{node_type}] {label_str}")
        else:
            raw_parts.append(str(node_id))
        if i < len(walk_edge_types):
            raw_parts.append(walk_edge_types[i])

    token_ids, node_boundaries, edge_boundaries = tokenizer.tokenize_walk(
        walk_nodes, walk_edge_types, ds_indexid2msg
    )
    tokens = [tokenizer.id2token.get(tid, f"<{tid}>") for tid in token_ids]

    tok_parts = []
    for i, (ns, ne) in enumerate(node_boundaries):
        tok_parts.append(" ".join(tokens[ns:ne]))
        if i < len(edge_boundaries):
            es, ee = edge_boundaries[i]
            tok_parts.append(" ".join(tokens[es:ee]))

    return " | ".join(raw_parts) + "\n" + " | ".join(tok_parts) + f" ({len(token_ids)})\n\n"


def _dump_walk_worker_mp(item):
    """Multiprocessing wrapper for _dump_walk_worker (uses global tokenizer)."""
    return _dump_walk_worker(item, _dump_tokenizer)


def _dump_corpus_to_txt(corpus, tokenizer, out_dir):
    """Write the walk corpus to a human-readable text file for inspection.

    For each walk, three lines are written:
        1. Original walk: node labels and edge types separated by ' | '
               [PROC] bash usr | write | [FILE] tmp foo | ...
        2. Tokenized walk with token count in parentheses:
               [CLS] [PROC] bash usr [SEP] | write | [FILE] tmp foo [SEP] | ... (42)
        3. Blank line

    Uses fork-based multiprocessing to parallelize tokenization across cores.
    """
    global _dump_tokenizer

    out_path = os.path.join(out_dir, "walk_corpus.txt")
    num_workers = min(mp.cpu_count(), 8)

    if num_workers <= 1 or len(corpus) < 1000:
        with open(out_path, "w") as f:
            for item in corpus:
                f.write(_dump_walk_worker(item, tokenizer))
    else:
        _dump_tokenizer = tokenizer
        ctx = mp.get_context("fork")
        chunksize = max(1, len(corpus) // (num_workers * 4))
        with open(out_path, "w", buffering=1 << 20) as f:
            with ctx.Pool(num_workers) as pool:
                for text in pool.imap(_dump_walk_worker_mp, corpus, chunksize=chunksize):
                    f.write(text)
        _dump_tokenizer = None

    log(f"Tokenized walk corpus written to {out_path} ({len(corpus):,} walks)")


def _dump_t5_corpus_to_txt(dump_data, tokenizer, out_dir):
    """Write deduped T5 corpus grouped by entity label.

    Each line: raw walk  ==>  tokenized walk
    """
    from pidsmaker.spider.data.tokenizer_bpe import NODE_TYPE_TOKENS

    out_path = os.path.join(out_dir, "walk_corpus.txt")
    id2tok = tokenizer.id2token

    # Group by (ntype, entity_label)
    grouped = defaultdict(list)
    for ntype, entity_label, direction, context_edges, context_nodes, ds_indexid2msg, decoder_ids in dump_data:
        key = (ntype, entity_label)
        grouped[key].append((direction, context_edges, context_nodes, ds_indexid2msg, decoder_ids))

    with open(out_path, "w", buffering=1 << 20) as f:
        for (ntype, entity_label), walks in grouped.items():
            type_tok = NODE_TYPE_TOKENS.get(ntype, f"[{ntype}]")
            header = f"{type_tok} {entity_label}".strip()
            f.write(f"=== {header} ===\n")
            for direction, context_edges, context_nodes, ds_indexid2msg, decoder_ids in walks:
                # Raw part
                raw_parts = [f"[{direction.upper()}]"]
                for i, nid in enumerate(context_nodes):
                    if i < len(context_edges):
                        raw_parts.append(context_edges[i])
                    if nid in ds_indexid2msg:
                        _n_type, n_label = ds_indexid2msg[nid]
                        type_prefix = NODE_TYPE_TOKENS.get(_n_type, f"[{_n_type}]")
                        raw_parts.append(f"{type_prefix} {n_label}")
                    else:
                        raw_parts.append(str(nid))
                raw_str = " | ".join(raw_parts)

                # Tokenized part
                toks = [id2tok.get(t, f"<{t}>") for t in decoder_ids]
                tok_str = " ".join(toks)

                f.write(f"{raw_str}  ==>  {tok_str} ({len(decoder_ids)})\n")
            f.write("\n")

    log(f"T5 walk corpus written to {out_path} ({len(dump_data):,} walks, "
        f"{len(grouped):,} entities)")

    # Write edge filter summary if stats are available
    filter_stats = getattr(_build_t5_parallel_edge_corpus, "filter_stats", None)
    if filter_stats and filter_stats["filtered"] > 0:
        _write_edge_filter_summary(filter_stats, out_dir)


def _write_edge_filter_summary(stats, out_dir):
    """Write a summary of filtered vs kept edges alongside the walk corpus."""
    out_path = os.path.join(out_dir, "edge_filter_summary.txt")

    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EDGE NOISE FILTER SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total edges enumerated:  {stats['total_edges']:>10,}\n")
        f.write(f"Edges filtered (noisy):  {stats['filtered']:>10,}  ({stats['filter_rate']:.1f}%)\n")
        f.write(f"Edges kept:              {stats['kept']:>10,}\n")
        f.write(f"Unique after dedup:      {stats['unique']:>10,}  ({stats['dedup_rate']:.1f}% dedup)\n")
        f.write("\n")

        # Skipped center entities
        skipped = stats["skipped_centers"]
        if skipped:
            f.write("-" * 70 + "\n")
            f.write(f"SKIPPED CENTER ENTITIES ({len(skipped):,} unique)\n")
            f.write("-" * 70 + "\n")
            for ctype, clabel in sorted(skipped):
                f.write(f"  [{ctype.upper()}] {clabel}\n")
            f.write("\n")

        # Filtered edges
        filtered = stats["filtered_examples"]
        if filtered:
            f.write("-" * 70 + "\n")
            f.write(f"FILTERED EDGES ({len(filtered):,} unique)\n")
            f.write("-" * 70 + "\n")
            for center, direction, edge_type, neighbor in sorted(filtered):
                f.write(f"  {center}  --[{direction} {edge_type}]-->  {neighbor}\n")
            f.write("\n")

    log(f"Edge filter summary written to {out_path} "
        f"({stats['filtered']:,} filtered, {len(skipped):,} skipped centers, "
        f"{len(filtered):,} unique filtered edges)")


def _dump_gnn_distill_corpus_to_txt(sampler_pairs, combined_indexid2msg, out_dir,
                                     filter_noisy=False):
    """Write GNN distillation neighborhood corpus to text file.

    Same format as the T5 walk_corpus.txt: each entity is a header, followed
    by its FORWARD and BACKWARD neighbors with edge types.

    Uses the full adjacency lists (all edges, not sampled) so the dump is
    deterministic and complete — useful for inspecting what the GNN sees.
    """
    from ..data.edge_filter import _is_noisy_file, _is_noisy_process
    from pidsmaker.spider.data.tokenizer_bpe import NODE_TYPE_TOKENS

    out_path = os.path.join(out_dir, "walk_corpus.txt")
    n_entities = 0
    n_edges_kept = 0
    n_edges_filtered = 0
    n_skipped_centers = 0

    # Group by (ntype, entity_label) for consistent ordering
    grouped = defaultdict(lambda: {"forward": [], "backward": []})
    seen_edges = defaultdict(set)  # key -> set of (direction, edge_type, neigh_tok, neigh_label)

    for sampler, ds_indexid2msg in sampler_pairs:
        for node in sampler.nodes:
            if node not in ds_indexid2msg:
                continue
            ntype, nlabel = ds_indexid2msg[node]

            # Skip noisy centers
            if filter_noisy:
                if ntype == "file" and _is_noisy_file(nlabel):
                    n_skipped_centers += 1
                    continue
                if ntype == "subject" and _is_noisy_process(nlabel):
                    n_skipped_centers += 1
                    continue

            key = (ntype, nlabel)

            for forward in (True, False):
                direction = "forward" if forward else "backward"
                adj = sampler.forward_adj if forward else sampler.backward_adj
                if node not in adj:
                    continue
                for neigh_node, edge_type, _ts in adj[node]:
                    if neigh_node not in ds_indexid2msg:
                        continue
                    neigh_type, neigh_label = ds_indexid2msg[neigh_node]

                    if filter_noisy:
                        if neigh_type == "file" and _is_noisy_file(neigh_label):
                            n_edges_filtered += 1
                            continue
                        if neigh_type == "subject" and _is_noisy_process(neigh_label):
                            n_edges_filtered += 1
                            continue

                    n_edges_kept += 1
                    type_tok = NODE_TYPE_TOKENS.get(neigh_type, f"[{neigh_type}]")
                    entry = (direction, edge_type, type_tok, neigh_label)
                    if entry not in seen_edges[key]:
                        seen_edges[key].add(entry)
                        grouped[key][direction].append(entry)

    with open(out_path, "w", buffering=1 << 20) as f:
        for (ntype, nlabel), directions in sorted(grouped.items()):
            type_tok = NODE_TYPE_TOKENS.get(ntype, f"[{ntype}]")
            f.write(f"=== {type_tok} {nlabel} ===\n")
            n_entities += 1
            for direction in ("backward", "forward"):
                for _dir, edge_type, neigh_tok, neigh_label in directions[direction]:
                    f.write(f"[{direction.upper()}] | {edge_type} | "
                            f"{neigh_tok} {neigh_label}\n")
            f.write("\n")

    filter_msg = ""
    if filter_noisy:
        total = n_edges_kept + n_edges_filtered
        rate = 100 * n_edges_filtered / total if total > 0 else 0
        filter_msg = (f", filtered {n_edges_filtered:,} noisy ({rate:.1f}%), "
                      f"skipped {n_skipped_centers:,} noisy centers")
    log(f"GNN distill corpus written to {out_path} "
        f"({n_entities:,} entities, {n_edges_kept:,} edges{filter_msg})")


def _eval_knn_on_eval_set(model, tokenizer, device, model_type, batch_size=512):
    """Compute KNN accuracy on the provenance eval set using the in-memory model.

    Returns dict with 'knn_accuracy_k5' and 'knn_accuracy_k5_per_cluster', or
    empty dict if eval set is unavailable.
    """
    import numpy as np

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..')
    eval_set_path = os.path.join(base_dir, 'config', 'eval', 'eval_set_clustering.json')
    if not os.path.exists(eval_set_path):
        return {}

    try:
        import json as _json
        with open(eval_set_path, 'r') as f:
            dataset = _json.load(f)
    except Exception:
        return {}

    EVAL_TYPE_MAP = {'PROC': 'subject', 'FILE': 'file', 'SOCK': 'netflow', 'NETFLOW': 'netflow'}

    def knn_accuracy_in_full_graph(X_all, labeled_indices, labels, k=5):
        from sklearn.neighbors import NearestNeighbors
        from collections import Counter
        labeled_set = set(labeled_indices)
        nn = NearestNeighbors(n_neighbors=min(200, len(X_all))).fit(X_all)
        correct = 0
        total = 0
        for i, idx in enumerate(labeled_indices):
            distances, indices = nn.kneighbors(X_all[idx:idx+1])
            voter_labels = []
            for j in indices[0]:
                if j in labeled_set and j != idx:
                    j_label_pos = labeled_indices.index(j)
                    voter_labels.append(labels[j_label_pos])
                if len(voter_labels) >= k:
                    break
            if not voter_labels:
                continue
            prediction = Counter(voter_labels).most_common(1)[0][0]
            if prediction == labels[i]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    def knn_accuracy_per_cluster(X_all, labeled_indices, labels, k=5):
        from sklearn.neighbors import NearestNeighbors
        from collections import Counter, defaultdict
        labeled_set = set(labeled_indices)
        nn = NearestNeighbors(n_neighbors=min(200, len(X_all))).fit(X_all)
        cluster_correct = defaultdict(int)
        cluster_total = defaultdict(int)
        for i, idx in enumerate(labeled_indices):
            distances, indices = nn.kneighbors(X_all[idx:idx+1])
            voter_labels = []
            for j in indices[0]:
                if j in labeled_set and j != idx:
                    j_label_pos = labeled_indices.index(j)
                    voter_labels.append(labels[j_label_pos])
                if len(voter_labels) >= k:
                    break
            if not voter_labels:
                continue
            true_label = labels[i]
            prediction = Counter(voter_labels).most_common(1)[0][0]
            cluster_total[true_label] += 1
            if prediction == true_label:
                cluster_correct[true_label] += 1
        return {
            cl: round(cluster_correct[cl] / cluster_total[cl], 4)
            for cl in cluster_total
        }

    # Tokenize all eval entities
    all_token_ids = []
    for entry in dataset:
        ntype = EVAL_TYPE_MAP.get(entry['entity_type'])
        if ntype is None:
            all_token_ids.append([tokenizer.pad_id])
            continue
        ids = tokenizer.tokenize_node(ntype, entry['entity_text'])
        if not ids:
            ids = [tokenizer.pad_id]
        all_token_ids.append(ids)

    # Encode in batches
    model.eval()
    all_embeddings = []
    is_encoder_only = model_type in ("gnn_distill", "behavior_cluster", "spider")

    with torch.no_grad():
        for batch_start in range(0, len(all_token_ids), batch_size):
            batch = all_token_ids[batch_start:batch_start + batch_size]
            input_ids, attention_mask = pad_token_id_lists(batch, tokenizer.pad_id, tokenizer.max_seq_len)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if is_encoder_only and hasattr(model, 'mean_pool'):
                hidden_states = model.mean_pool(input_ids, attention_mask)
                mean_pooled = hidden_states.cpu().numpy()
            elif is_encoder_only and hasattr(model, 'encode'):
                mean_pooled = model.encode(input_ids, attention_mask).cpu().numpy()
            else:
                hidden_states = model.modified_fwd(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=torch.full_like(input_ids, -100), skip_cls=True)
                mask_expanded = attention_mask.unsqueeze(-1).float()
                mean_pooled = ((hidden_states * mask_expanded).sum(dim=1) /
                               mask_expanded.sum(dim=1).clamp(min=1)).cpu().numpy()

            all_embeddings.append(mean_pooled)

    emb = np.concatenate(all_embeddings, axis=0)
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb = emb / norms

    clusters = [e['cluster'] for e in dataset]
    all_indices = list(range(len(emb)))

    knn_acc = knn_accuracy_in_full_graph(emb, all_indices, clusters, k=5)
    knn_per_cluster = knn_accuracy_per_cluster(emb, all_indices, clusters, k=5)

    # ── Additional clustering & ranking metrics ───────────────────
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)
    cluster_to_int = {c: i for i, c in enumerate(unique_clusters)}
    labels_true = np.array([cluster_to_int[c] for c in clusters])

    # K-Means clustering → ARI & NMI
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(emb)
    labels_pred = km.labels_
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    # Silhouette score (using ground-truth labels)
    sil = silhouette_score(emb, labels_true)

    # Mean Average Precision: for each entity, rank all others by
    # embedding distance and compute AP against same-cluster members
    sim_matrix = emb @ emb.T  # cosine sim (already L2-normed)
    n = len(emb)
    ap_scores = []
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        ranked = np.argsort(-sims)
        relevant = (labels_true[ranked] == labels_true[i])
        n_relevant = relevant.sum()
        if n_relevant == 0:
            continue
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / (np.arange(len(ranked)) + 1)
        ap_scores.append((precision_at_k * relevant).sum() / n_relevant)
    mean_ap = float(np.mean(ap_scores)) if ap_scores else 0.0

    return {
        'knn_accuracy_k5': round(knn_acc, 4),
        'knn_accuracy_k5_per_cluster': knn_per_cluster,
        'ari': round(ari, 4),
        'nmi': round(nmi, 4),
        'silhouette': round(sil, 4),
        'mAP': round(mean_ap, 4),
    }


def _load_dataset_for_pretrain(cfg, dataset_name):
    """Load indexid2msg, train nodes, and graph paths for a dataset."""
    if dataset_name == cfg.dataset.name:
        ds_cfg = cfg
    else:
        ds_cfg = deepcopy(cfg)
        set_dataset_cfg(ds_cfg, dataset_name)
        set_task_paths(ds_cfg)

    indexid2msg = get_indexid2msg(ds_cfg)

    splits = get_splits_to_train_featurization(ds_cfg)
    split2nodes = get_split2nodes(ds_cfg)
    train_nodes = set().union(*(split2nodes[split] for split in splits))

    base_dir = ds_cfg.preprocessing.transformation._graphs_dir
    train_paths = get_all_files_from_folders(base_dir, ds_cfg.dataset.train_files)
    val_paths = get_all_files_from_folders(base_dir, ds_cfg.dataset.val_files)

    return indexid2msg, train_nodes, train_paths, val_paths
