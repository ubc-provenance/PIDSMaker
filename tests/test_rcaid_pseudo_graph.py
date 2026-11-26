"""Pseudo-graph of R-Caid as described in the paper (Goyal et al., IEEE S&P 2024), and its
bounds. Runs without database."""

import collections
import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from pidsmaker.preprocessing.transformation_methods import transformation_rcaid_pseudo_graph as T
from pidsmaker.utils.graph_store import load_graph_columns, save_graph

from test_rcaid_transformation import _random_graph, _signature

pytestmark = pytest.mark.no_database


def _cfg(use_pruning=True, prune_threshold=0.5, root_node_types="file, netflow",
         keep_original_edges=True, merge_same_label_roots=True, max_roots_per_node=0):
    return SimpleNamespace(
        transformation=SimpleNamespace(
            rcaid_pseudo_graph=SimpleNamespace(
                use_pruning=use_pruning,
                prune_threshold=prune_threshold,
                root_node_types=root_node_types,
                keep_original_edges=keep_original_edges,
                merge_same_label_roots=merge_same_label_roots,
                max_roots_per_node=max_roots_per_node,
            )
        )
    )


def _labelled_graph(seed, num_nodes=60, num_edges=150):
    """Random graph whose labels repeat, so that roots share labels."""
    G = _random_graph(seed, num_nodes, num_edges)
    rng = random.Random(seed)
    for n, d in G.nodes(data=True):
        d["label"] = rng.choice(["/lib/a", "/lib/b", "None", f"/x/{n}"])
    for _, _, d in G.edges(data=True):
        d["y"] = 0
    return G


def _reference_pseudo_edges(G, cfg):
    """(root, descendant) pairs from the definitions, with networkx."""
    c = cfg.transformation.rcaid_pseudo_graph
    types = None if c.root_node_types == "all" else {t.strip() for t in c.root_node_types.split(",")}
    first_out, first_in = {}, {}
    for u, v, d in G.edges(data=True):
        first_out[u] = min(first_out.get(u, d["time"]), d["time"])
        first_in[v] = min(first_in.get(v, d["time"]), d["time"])
    kept = []  # (root, descendants), roots in node order
    for r in G.nodes:
        if r not in first_out or (r in first_in and first_in[r] <= first_out[r]):
            continue
        if types is not None and G.nodes[r]["node_type"] not in types:
            continue
        desc = nx.descendants(G, r)
        if c.use_pruning and len(desc) > c.prune_threshold * G.number_of_nodes():
            continue
        kept.append((r, desc))
    if c.max_roots_per_node:
        # Roots with the fewest descendants first (stable: then in node order)
        kept.sort(key=lambda rd: len(rd[1]))
    pairs, seen, count = [], set(), collections.Counter()
    for r, desc in kept:
        key = (G.nodes[r]["node_type"], G.nodes[r]["label"])
        for d in desc:
            if c.merge_same_label_roots:
                if (key, d) in seen:
                    continue
            if c.max_roots_per_node and count[d] >= c.max_roots_per_node:
                continue
            seen.add((key, d))
            count[d] += 1
            pairs.append((r, d))
    return pairs


def _split_edges(H):
    orig, pseudo = [], []
    for u, v, d in H.edges(data=True):
        (pseudo if d["label"] is None else orig).append((u, v, d))
    return orig, pseudo


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("threshold", [0.5, 0.1])
@pytest.mark.parametrize("types", ["file, netflow", "all"])
@pytest.mark.parametrize("max_roots", [0, 1, 3])
def test_pseudo_graph_matches_definition(tmp_path, seed, merge, threshold, types, max_roots):
    G = _labelled_graph(seed)
    cfg = _cfg(prune_threshold=threshold, root_node_types=types, merge_same_label_roots=merge,
               max_roots_per_node=max_roots)
    np.random.seed(0)
    H = T.main(G.copy(), cfg)
    orig, pseudo = _split_edges(H)

    # The original edges are kept with their attributes, the nodes too
    assert list(H.nodes(data=True)) == list(G.nodes(data=True))
    assert sorted((u, v, sorted(d.items())) for u, v, d in orig) == sorted(
        (u, v, sorted(d.items())) for u, v, d in G.edges(data=True)
    )
    # Pseudo edges: no type, y of 0, a time in the range of the graph
    times = [d["time"] for _, _, d in G.edges(data=True)]
    for _, _, d in pseudo:
        assert d["y"] == 0 and min(times) <= d["time"] < max(times)
    ref = _reference_pseudo_edges(G, cfg)
    assert sorted((u, v) for u, v, _ in pseudo) == sorted(ref)
    if max_roots:
        assert max(collections.Counter(v for _, v, _ in pseudo).values(), default=0) <= max_roots

    # Same graph when saved without building it, from the graph and from its columns
    ref_sig = _signature(H)
    np.random.seed(0)
    T.transform_and_save(G.copy(), cfg, tmp_path / "a.pt")
    assert _signature(torch.load(tmp_path / "a.pt")) == ref_sig
    save_graph(G, tmp_path / "in.pt")
    np.random.seed(0)
    T.transform_columns_and_save(load_graph_columns(tmp_path / "in.pt"), cfg, tmp_path / "b.pt")
    assert _signature(torch.load(tmp_path / "b.pt")) == ref_sig


def test_only_pseudo_edges_without_original_edges():
    G = _labelled_graph(1)
    cfg = _cfg(keep_original_edges=False, merge_same_label_roots=False, root_node_types="all")
    H = T.main(G.copy(), cfg)
    assert sorted(H.edges()) == sorted(_reference_pseudo_edges(G, cfg))


def test_merge_same_label_roots_example():
    # Three files read by p: two with the same label merge into one pseudo edge per descendant
    G = nx.MultiDiGraph()
    for n, t, l in [("f1", "file", "/lib/a"), ("f2", "file", "/lib/a"), ("f3", "file", "/lib/b"),
                    ("p", "subject", "/bin/sh"), ("q", "subject", "/bin/ls")]:
        G.add_node(n, node_type=t, label=l)
    for i, (u, v) in enumerate([("f1", "p"), ("f2", "p"), ("f3", "p"), ("p", "q")]):
        G.add_edge(u, v, time=i, label="EVENT_READ", y=0)
    H = T.main(G, _cfg())
    _, pseudo = _split_edges(H)
    assert sorted((u, v) for u, v, _ in pseudo) == [("f1", "p"), ("f1", "q"), ("f3", "p"), ("f3", "q")]
    H = T.main(G, _cfg(root_node_types="subject"))  # p has an incoming edge first: no root
    assert not _split_edges(H)[1]
    # One root per node: all the roots reach {p, q}, so the first one in node order
    H = T.main(G, _cfg(max_roots_per_node=1))
    assert sorted((u, v) for u, v, _ in _split_edges(H)[1]) == [("f1", "p"), ("f1", "q")]
