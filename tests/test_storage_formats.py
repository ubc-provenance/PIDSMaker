"""Round-trip tests of the compact storage formats (graphs and loss files)."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from pidsmaker.utils.graph_store import (
    encode_graph,
    load_graph,
    load_graph_columns,
    save_graph,
)
from pidsmaker.utils.loss_files import read_losses, write_losses


def _signature(g):
    """Everything observable on a graph: values, Python types and all iteration orders."""

    def typed(d):
        return [(k, type(v).__name__, v) for k, v in d.items()]

    out = [type(g).__name__, typed(g.graph)]
    out.append([(type(n).__name__, n, typed(d)) for n, d in g._node.items()])
    if not g.is_directed():
        return out + [("E", list(g.edges(data=True)))]
    for u, nbrs in g._succ.items():
        for v, inner in nbrs.items():
            if g.is_multigraph():
                out.append(("S", u, v, [(type(k).__name__, k, typed(d)) for k, d in inner.items()]))
            else:
                out.append(("S", u, v, typed(inner)))
    for v, preds in g._pred.items():
        out.append(("P", v, list(preds)))
        for u, inner in preds.items():
            assert inner is g._succ[u][v]
    return out


def _roundtrip(g, tmp_path):
    path = str(tmp_path / "2018-04-06 00:00:00.000000000~2018-04-06 00:15:00.000000000")
    save_graph(g, path)
    h = load_graph(path)
    assert _signature(g) == _signature(h)
    assert torch.load(path).__class__ is g.__class__
    return path


def _provenance_graph():
    g = nx.MultiDiGraph()
    g.add_nodes_from(
        (str(i), {"node_type": t, "label": f"/usr/bin/p{i % 3}"})
        for i, t in zip(range(10, 0, -1), ["subject", "file", "netflow"] * 4)
    )
    edges = [("3", "1"), ("1", "3"), ("3", "1"), ("2", "3"), ("10", "1"), ("1", "2"), ("3", "1")]
    g.add_edges_from(
        (u, v, {"event_uuid": f"8A1F0C3E-0000-4000-8000-{i:012X}", "time": 10**18 + i, "label": "EVENT_READ", "y": 0})
        for i, (u, v) in enumerate(edges)
    )
    return g


def test_provenance_graph_is_compact_and_identical(tmp_path):
    g = _provenance_graph()
    assert encode_graph(g) is not None
    path = _roundtrip(g, tmp_path)
    cols = load_graph_columns(path)
    assert cols.node_ids == list(g.nodes)
    assert [(cols.node_ids[s], cols.node_ids[d]) for s, d in zip(cols.src, cols.dst)] == list(g.edges())
    assert cols.edge_attrs["event_uuid"] == [d["event_uuid"] for *_, d in g.edges(data=True)]


def test_predecessor_order_and_removals(tmp_path):
    g = _provenance_graph()
    g.remove_edge("3", "1", key=1)
    g.remove_node("2")
    g.add_edge("5", "1", key="custom", event_uuid="x", time=1, label="L", y=1)
    g.add_edge("4", "4", event_uuid="self-loop", time=2, label="L", y=0)
    _roundtrip(g, tmp_path)


def test_digraph_and_mixed_values(tmp_path):
    g = nx.DiGraph(name="g")
    g.add_node(1, w=0.5, tag=None)
    g.add_node(2, w=1.5, tag=np.int64(3))
    g.add_edge(2, 1, ts=2**70, label="lowercase-not-uuid-0000-000000000000")
    g.add_edge(1, 2, ts=1, label="é")
    _roundtrip(g, tmp_path)


@pytest.mark.parametrize(
    "graph",
    [
        nx.MultiDiGraph(),
        nx.Graph([(1, 2)]),
        nx.MultiGraph([(1, 2), (1, 2)]),
    ],
)
def test_empty_and_unsupported_graphs(graph, tmp_path):
    _roundtrip(graph, tmp_path)


def test_heterogeneous_attributes_fall_back_to_pickle(tmp_path):
    g = _provenance_graph()
    g.nodes["1"]["extra"] = 1
    assert encode_graph(g) is None
    path = _roundtrip(g, tmp_path)
    assert load_graph_columns(path) is None


def test_string_encodings(tmp_path):
    g = nx.MultiDiGraph()
    g.add_node("007", node_type="file", label="a")  # not a canonical integer
    g.add_node("7", node_type="file", label="b")
    g.add_edge("007", "7", event_uuid="8a1f0c3e-0000-4000-8000-000000000001", time=1, label="x", y=0)
    g.add_edge("7", "007", event_uuid="8a1f0c3e-0000-4000-8000-000000000002", time=2, label="y", y=0)
    _roundtrip(g, tmp_path)


def test_loss_files_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "loss": np.random.rand(100).astype(np.float32).astype(float),
            "score": np.random.rand(100),  # not float32-representable
            "srcnode": np.arange(100),
            "time": np.arange(100) + 1_522_987_200_017_083_676,
            "flag": np.random.rand(100) > 0.5,
        }
    )
    write_losses(str(tmp_path), "2018-04-06 00:00:00.0~2018-04-06 00:15:00.0", df)
    df.to_csv(tmp_path / "legacy.csv", index=False)
    back = read_losses(str(tmp_path / "2018-04-06 00:00:00.0~2018-04-06 00:15:00.0.npz"))
    legacy = read_losses(str(tmp_path / "legacy.csv"))
    expected = df.assign(flag=df["flag"].astype(np.int64))
    pd.testing.assert_frame_equal(back, expected)
    pd.testing.assert_frame_equal(legacy[["loss", "srcnode", "time"]], expected[["loss", "srcnode", "time"]])
    sub = read_losses(str(tmp_path / "2018-04-06 00:00:00.0~2018-04-06 00:15:00.0.npz"), usecols=["loss"])
    assert list(sub.columns) == ["loss"]
