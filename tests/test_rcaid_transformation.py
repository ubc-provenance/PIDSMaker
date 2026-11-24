"""The R-Caid pseudo-graph transformation gives the same graphs as its original implementation
(kept below), on small synthetic graphs. Runs without a database:

    python -m pytest tests/test_rcaid_transformation.py -k rcaid
"""

import os
import random
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from pidsmaker.preprocessing.transformation_methods import transformation_rcaid_pseudo_graph
from pidsmaker.utils.graph_store import load_graph_columns, save_graph

pytestmark = pytest.mark.no_database


# --------------------------------------------------------------------------------------
# Original implementation
# --------------------------------------------------------------------------------------


def _original_add_arbitrary_timestamps_to_graph(original_G, new_G):
    timestamps = [edge[2]["time"] for edge in original_G.edges(data=True)]
    min_t, max_t = min(timestamps), max(timestamps)
    rand_t = np.random.randint(min_t, max_t, size=len(new_G.edges()))

    for t, (src, dst) in zip(rand_t, new_G.edges()):
        new_G[src][dst]["time"] = t

    return new_G


def _original_identify_root_nodes(G):
    root_nodes = set()

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        in_edges = list(G.in_edges(node, data=True))

        earliest_out_time = min(edge[2]["time"] for edge in out_edges) if out_edges else None
        earliest_in_time = min(edge[2]["time"] for edge in in_edges) if in_edges else None

        if earliest_out_time is not None and (
            earliest_in_time is None or earliest_out_time < earliest_in_time
        ):
            root_nodes.add(node)

    return root_nodes


def _original_create_pseudo_graph(G, root_nodes):
    pseudo_graph = nx.DiGraph()
    pseudo_to_root = {}

    for node, attr in G.nodes(data=True):
        pseudo_graph.add_node(node, **attr)

    for root in root_nodes:
        pseudo_root = f"pseudo_{root}"
        pseudo_to_root[pseudo_root] = root
        pseudo_graph.add_node(pseudo_root, **G.nodes[root])

        descendants = nx.descendants(G, root)
        for descendant in descendants:
            pseudo_graph.add_edge(pseudo_root, descendant)

    return pseudo_graph, pseudo_to_root


def _original_prune_pseudo_roots(pseudo_graph, G, prune_threshold, pseudo_roots):
    total_nodes_in_G = len(G.nodes())
    max_allowed_connections = prune_threshold * total_nodes_in_G

    pseudo_roots_to_prune = []
    for node in pseudo_roots:
        if node not in pseudo_graph:
            continue
        num_connections = len(list(pseudo_graph.successors(node)))
        if num_connections > max_allowed_connections:
            pseudo_roots_to_prune.append(node)

    for pseudo_root in pseudo_roots_to_prune:
        pseudo_graph.remove_node(pseudo_root)

    return pseudo_graph


def original_rcaid(graph, cfg):
    root_nodes = _original_identify_root_nodes(graph)
    pseudo_graph, pseudo_to_root = _original_create_pseudo_graph(graph, root_nodes)

    if cfg.transformation.rcaid_pseudo_graph.use_pruning:
        pseudo_graph = _original_prune_pseudo_roots(
            pseudo_graph, graph, 0.5, list(pseudo_to_root.keys())
        )

    pseudo_graph = nx.relabel_nodes(pseudo_graph, pseudo_to_root)
    pseudo_graph = _original_add_arbitrary_timestamps_to_graph(original_G=graph, new_G=pseudo_graph)
    return nx.MultiDiGraph(pseudo_graph)


# --------------------------------------------------------------------------------------
# Synthetic graphs
# --------------------------------------------------------------------------------------


def _cfg(use_pruning):
    return SimpleNamespace(
        transformation=SimpleNamespace(rcaid_pseudo_graph=SimpleNamespace(use_pruning=use_pruning))
    )


def _random_graph(seed, num_nodes, num_edges, int_ids=False, multi=True, back=0.03, max_time=1000):
    """Random provenance-like graph: mostly forward edges in a random node order with a few
    backward ones (cycles), self-loops, parallel edges, hubs, and extra roots sharing the same
    successors (like files read by the same processes)."""
    rng = random.Random(seed)
    G = nx.MultiDiGraph() if multi else nx.DiGraph()
    ids = list(range(num_nodes + num_nodes // 4))
    rng.shuffle(ids)
    ids = [i * 7 + 3 if int_ids else f"{i * 7 + 3}" for i in ids]
    for i in ids:
        G.add_node(i, node_type=rng.choice(["subject", "file", "netflow"]), label=f"n{i}")
    hubs = list(range(0, num_nodes, 10))
    for _ in range(num_edges):
        a = rng.randrange(num_nodes)
        b = rng.choice(hubs) if rng.random() < 0.3 else rng.randrange(num_nodes)
        if rng.random() < back or a == b:
            u, v = max(a, b), min(a, b)
        else:
            u, v = min(a, b), max(a, b)
        G.add_edge(ids[u], ids[v], time=rng.randrange(max_time), label="EVENT")
    for i in range(num_nodes, len(ids)):
        for hub in rng.sample(hubs[:3], min(len(hubs[:3]), rng.choice([1, 1, 2]))):
            G.add_edge(ids[i], ids[hub], time=rng.randrange(max_time), label="EVENT")
    return G


def _signature(g):
    """Everything observable on a graph: values, Python types and all iteration orders."""

    def typed(d):
        return [(k, type(v).__name__, v) for k, v in d.items()]

    out = [type(g).__name__, typed(g.graph)]
    out.append([(type(n).__name__, n, typed(d)) for n, d in g._node.items()])
    for u, nbrs in g._succ.items():
        for v, inner in nbrs.items():
            out.append(("S", u, v, [(type(k).__name__, k, typed(d)) for k, d in inner.items()]))
    for v, preds in g._pred.items():
        out.append(("P", v, list(preds)))
        for u, inner in preds.items():
            assert inner is g._succ[u][v]
    out.append(list(g.nodes(data=True)))
    out.append(list(g.edges(keys=True, data=True)))
    return out


def _check_same(G, use_pruning, tmp_path, seed=0):
    cfg = _cfg(use_pruning)

    np.random.seed(seed)
    expected = original_rcaid(G.copy(), cfg)
    expected_state = np.random.get_state()[1].copy()
    ref = _signature(expected)

    np.random.seed(seed)
    got = transformation_rcaid_pseudo_graph.main(G.copy(), cfg)
    assert _signature(got) == ref
    assert np.array_equal(np.random.get_state()[1], expected_state)

    # Saved without building the result, from the graph and from its stored columns
    np.random.seed(seed)
    transformation_rcaid_pseudo_graph.transform_and_save(G.copy(), cfg, tmp_path / "a.pt")
    assert _signature(torch.load(tmp_path / "a.pt")) == ref

    save_graph(G, tmp_path / "in.pt")
    cols = load_graph_columns(tmp_path / "in.pt")
    if cols is not None:
        np.random.seed(seed)
        transformation_rcaid_pseudo_graph.transform_columns_and_save(cols, cfg, tmp_path / "b.pt")
        assert _signature(torch.load(tmp_path / "b.pt")) == ref
    return expected


@pytest.mark.parametrize("use_pruning", [True, False])
@pytest.mark.parametrize("int_ids", [False, True])
@pytest.mark.parametrize("density", [2, 2.5, 3])
@pytest.mark.parametrize("seed", range(6))
def test_rcaid_random_graphs(tmp_path, seed, density, int_ids, use_pruning):
    num_nodes = [8, 30, 60, 120, 200, 400][seed]
    G = _random_graph(seed, num_nodes, int(num_nodes * density), int_ids=int_ids)
    _check_same(G, use_pruning, tmp_path, seed)


@pytest.mark.parametrize("use_pruning", [True, False])
def test_rcaid_digraph_input(tmp_path, use_pruning):
    _check_same(_random_graph(1, 80, 200, multi=False), use_pruning, tmp_path)


@pytest.mark.parametrize("use_pruning", [True, False])
def test_rcaid_overlapping_descendants_and_cycles(tmp_path, use_pruning):
    # Roots 0, 1 and 2 share successors; 10 -> 11 -> 12 -> 10 is a cycle; root 20 is in a cycle
    # with its successor (20 -> 21 at t=1, 21 -> 20 at t=5); 30 has a self-loop
    G = nx.MultiDiGraph()
    for n in [0, 1, 2, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40]:
        G.add_node(n, node_type="file")
    edges = [
        (0, 10, 1), (1, 10, 2), (2, 10, 3), (2, 11, 3), (1, 10, 9),
        (10, 11, 4), (11, 12, 5), (12, 10, 6), (12, 13, 7),
        (20, 21, 1), (21, 20, 5), (21, 22, 6), (22, 13, 8),
        (30, 30, 1), (30, 31, 2), (31, 0, 3),
        (40, 0, 9), (13, 40, 10),
    ]
    for u, v, t in edges:
        G.add_edge(u, v, time=t)
    out = _check_same(G, use_pruning, tmp_path)
    if not use_pruning:
        assert out.out_degree(20) > 0  # root in a cycle with its successor


def test_rcaid_pruning_threshold(tmp_path):
    # A chain where the first nodes reach more than half of the nodes, and exactly half
    n = 10
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1, time=i)
    for i in range(0, n - 1, 2):
        G.add_edge(100 + i, i, time=-1)  # extra roots entering the chain
    for use_pruning in (True, False):
        _check_same(G, use_pruning, tmp_path)


def test_rcaid_float_times(tmp_path):
    G = _random_graph(3, 50, 120)
    for _, _, d in G.edges(data=True):
        d["time"] = float(d["time"]) + 0.5
    _check_same(G, True, tmp_path)


def test_rcaid_mixed_time_types(tmp_path):
    G = _random_graph(4, 50, 120)
    for i, (_, _, d) in enumerate(G.edges(data=True)):
        if i % 3 == 0:
            d["time"] = float(d["time"])
    _check_same(G, True, tmp_path)


def test_rcaid_no_edges():
    G = nx.MultiDiGraph()
    G.add_nodes_from(["a", "b"])
    with pytest.raises(ValueError):
        original_rcaid(G.copy(), _cfg(True))
    with pytest.raises(ValueError):
        transformation_rcaid_pseudo_graph.main(G.copy(), _cfg(True))


# --------------------------------------------------------------------------------------
# Transformation task: windows transformed in parallel
# --------------------------------------------------------------------------------------


def _task_cfg(num_workers, dates):
    return SimpleNamespace(
        _num_workers=num_workers,
        dataset=SimpleNamespace(train_dates=dates[:1], val_dates=dates[1:2], test_dates=dates[2:]),
        transformation=SimpleNamespace(rcaid_pseudo_graph=SimpleNamespace(use_pruning=True)),
    )


def _force_parallel(monkeypatch):
    """Lets the task fork its workers (torch_geometric may initialize CUDA when imported by
    other tests, in which case the task does not fork) and returns its log messages."""
    from pidsmaker.tasks import transformation

    messages = []
    monkeypatch.setattr(transformation, "_can_fork", lambda: True)
    monkeypatch.setattr(transformation, "log", lambda msg, *a, **k: messages.append(msg))
    return messages


@pytest.mark.parametrize("from_columns", [True, False])
def test_rcaid_task_parallel_matches_sequential(tmp_path, monkeypatch, from_columns):
    """Windows transformed in parallel are identical to a sequential run of the original
    implementation, including the random timestamps drawn in window order and the final random
    state."""
    from pidsmaker.tasks import transformation

    dates = ["2018-04-01", "2018-04-02", "2018-04-03"]
    base = tmp_path / "in"
    graphs = []
    for d, date in enumerate(dates):
        os.makedirs(base / f"graph_{date}")
        for w in range(3):
            G = _random_graph(10 * d + w, [40, 150, 300][w], [80, 360, 700][w])
            path = base / f"graph_{date}" / f"{date} 0{w}:00:00~{date} 0{w}:15:00"
            if from_columns:
                save_graph(G, path)
            else:
                torch.save(G, path)  # legacy pickled graph: loaded as networkx
            graphs.append((str(path), G))

    messages = _force_parallel(monkeypatch)
    cfg = _task_cfg(3, dates)
    np.random.seed(0)
    expected = {}
    for path, G in sorted(graphs, key=lambda x: (x[0].split("/")[-2], x[0])):
        expected[path.split("/")[-1]] = _signature(original_rcaid(G, cfg))
    expected_state = np.random.get_state()[1].copy()

    for num_workers in (1, 3):
        out = tmp_path / f"out{num_workers}"
        np.random.seed(0)
        transformation.add_graph_transformation(
            str(base) + "/", str(out) + "/", _task_cfg(num_workers, dates), ["rcaid_pseudo_graph"]
        )
        assert np.array_equal(np.random.get_state()[1], expected_state)
        assert any("3 processes" in m for m in messages) == (num_workers == 3)
        files = [p for p in out.glob("*/*")]
        assert len(files) == len(expected)
        for p in files:
            assert _signature(torch.load(p)) == expected[p.name], (num_workers, p.name)


def test_rcaid_task_parallel_error(tmp_path, monkeypatch):
    """A failing window stops the parallel transformation with its error."""
    from pidsmaker.tasks import transformation

    messages = _force_parallel(monkeypatch)
    dates = ["2018-04-01", "2018-04-02", "2018-04-03"]
    for d, date in enumerate(dates):
        os.makedirs(tmp_path / "in" / f"graph_{date}")
        for w in range(2):
            G = _random_graph(d, 50, 100)
            if d == 1 and w == 1:
                G.remove_edges_from(list(G.edges()))  # no timestamps: ValueError
            save_graph(G, tmp_path / "in" / f"graph_{date}" / f"{date} 0{w}:00:00~{date} 0{w}:15:00")
    with pytest.raises(ValueError):
        transformation.add_graph_transformation(
            str(tmp_path / "in") + "/", str(tmp_path / "out") + "/", _task_cfg(3, dates),
            ["rcaid_pseudo_graph"],
        )
    assert any("3 processes" in m for m in messages)
