"""Round-trip tests of the compact storage formats (graphs and loss files)."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from pidsmaker.utils.graph_store import (
    encode_attr_dicts,
    encode_edge_positions,
    encode_graph,
    load_graph,
    load_graph_columns,
    save_graph,
)
from pidsmaker.utils.loss_files import read_losses, write_losses

pytestmark = pytest.mark.no_database


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


def test_loss_file_replaces_stale_csv_of_same_window(tmp_path):
    name = "2018-04-06 00:00:00.0~2018-04-06 00:15:00.0"
    pd.DataFrame({"loss": [1.0], "srcnode": [1]}).to_csv(tmp_path / f"{name}.csv", index=False)
    write_losses(str(tmp_path), name, {"loss": np.array([2.0]), "srcnode": np.array([1])})
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{name}.npz"]


def test_edge_messages_ignore_isolated_nodes():
    """Featurizations that embed only nodes with edges (flash, alacarte) have no vector
    for isolated nodes, which must not be looked up (as before the vectorization)."""
    from pidsmaker.tasks.feat_inference import _edge_columns_from_graph, _vectorized_edge_msg
    from pidsmaker.utils.data_utils import CollatableTemporalData, expand_compact_msg

    g = nx.MultiDiGraph()
    g.add_node("1", node_type="subject", label="a")
    g.add_node("3", node_type="file", label="isolated")
    g.add_node("2", node_type="file", label="b")
    g.add_edge("1", "2", time=5, label="EVENT_READ", y=0)
    g.add_edge("2", "1", time=6, label="EVENT_WRITE", y=1)
    indexid2vec = {"1": np.array([1.0, 2.0]), "2": np.array([3.0, 4.0])}
    ntype2oh = {t: torch.eye(3)[i] for i, t in enumerate(["subject", "file", "netflow"])}
    etype2oh = {t: torch.eye(2)[i] for i, t in enumerate(["EVENT_READ", "EVENT_WRITE"])}

    compact, src, dst, t, y = _vectorized_edge_msg(
        _edge_columns_from_graph(g), indexid2vec, etype2oh, ntype2oh
    )
    assert (src, dst, t, y) == ([1, 2], [2, 1], [5, 6], [0, 1])
    data = CollatableTemporalData(src=torch.tensor(src), dst=torch.tensor(dst), t=torch.tensor(t))
    for k, v in compact.items():
        data[k] = v
    msg = expand_compact_msg(data).msg
    feats = {n: torch.cat([ntype2oh[g.nodes[n]["node_type"]], torch.tensor(indexid2vec[n]).float()]) for n in ["1", "2"]}
    expected = torch.stack([
        torch.cat([feats["1"], etype2oh["EVENT_READ"], feats["2"]]),
        torch.cat([feats["2"], etype2oh["EVENT_WRITE"], feats["1"]]),
    ])
    assert torch.equal(msg, expected)


def _payload_bytes(payload):
    import pickle

    return {k: pickle.dumps(v) for k, v in payload.items()}


@pytest.mark.parametrize("num_edges", [0, 1, 200])
def test_edge_list_payload_matches_graph(num_edges):
    """`encode_edge_list` gives the payload of the graph built from the same edges."""
    from pidsmaker.utils.graph_store import encode_edge_list

    rng = np.random.default_rng(num_edges)
    nodes = [str(i) for i in rng.permutation(30)]
    node_attrs = {"node_type": [f"t{i % 3}" for i in range(30)], "label": [f"n{i}" for i in range(30)]}
    # Few nodes: repeated pairs, self-loops and interleaved pairs
    src = [nodes[i] for i in rng.integers(0, 8, num_edges)]
    dst = [nodes[i] for i in rng.integers(0, 8, num_edges)]
    edge_attrs = {
        "event_uuid": [f"8A1F0C3E-0000-4000-8000-{i:012X}" for i in range(num_edges)],
        "time": [int(t) for t in rng.integers(0, 50, num_edges)],
        "label": [f"EVENT_{i % 4}" for i in range(num_edges)],
        "y": [int(i % 7 == 0) for i in range(num_edges)],
        "size": [float(i) / 3 for i in range(num_edges)],
    }
    g = nx.MultiDiGraph()
    g.add_nodes_from((n, {k: v[i] for k, v in node_attrs.items()}) for i, n in enumerate(nodes))
    g.add_edges_from(
        (u, v, {k: col[i] for k, col in edge_attrs.items()}) for i, (u, v) in enumerate(zip(src, dst))
    )
    expected = _payload_bytes(encode_graph(g))
    assert _payload_bytes(encode_edge_list(nodes, node_attrs, src, dst, edge_attrs)) == expected


def test_lazy_columns_match_decoded_columns(tmp_path):
    """`take`, `factorize` and `array` of the columns of a stored graph match its values."""
    g = nx.MultiDiGraph()
    g.add_nodes_from([("1", {"t": "a"}), ("2", {"t": "b"})])
    values = {
        "int": [5, 3, 5, 9],
        "float": [0.5, float("nan"), 0.5, 2.0],
        "strdict": ["x", "y", "x", "x"],
        "uuid": [f"8A1F0C3E-0000-4000-8000-{i:012X}" for i in range(4)],
        "intstr": ["10", "20", "30", "40"],
        "strblob": ["a", "bb", "ccc", "dddd"],
        "obj": [None, 1, "s", 2.5],
        "npint64": [np.int64(5), np.int64(3), np.int64(5), np.int64(9)],
    }
    for i in range(4):
        g.add_edge("1", "2", **{k: v[i] for k, v in values.items()})
    path = str(tmp_path / "g")
    save_graph(g, path)
    cols = load_graph_columns(path).edge_attrs
    for k, v in values.items():
        assert cols.take(k, [3, 0]) == [v[3], v[0]], k
        codes, uniques = cols.factorize(k)
        rebuilt = [uniques[c] for c in codes]
        assert all(a == b or (a != a and b != b) for a, b in zip(rebuilt, v)), k
        assert cols[k] == v or k == "float"
    assert cols.array("int", np.int64).tolist() == values["int"]
    assert cols.array("npint64", np.int64).tolist() == [5, 3, 5, 9]
    assert {type(v) for v in cols["npint64"] + cols.take("npint64", [1]) + cols.factorize("npint64")[1]} == {np.int64}


def test_numpy_int_attributes(tmp_path):
    """np.int64 values (e.g. random timestamps) are stored as an array and loaded as np.int64."""
    g = _provenance_graph()
    for i, (*_, d) in enumerate(g.edges(data=True)):
        d["time"] = np.int64(10**18 + i)
    assert encode_graph(g)["edge_attr_cols"][1][0] == "npint64"
    _roundtrip(g, tmp_path)


@pytest.mark.parametrize("num_edges", [0, 1, 200])
def test_edge_positions_payload_matches_graph(num_edges):
    """`encode_edge_positions` encodes the graph built by adding the edges in order."""
    rng = np.random.default_rng(num_edges)
    nodes = [str(i) for i in rng.permutation(60)]
    pairs = sorted({(int(u), int(v)) for u, v in rng.integers(0, 60, size=(num_edges, 2))})
    src = np.array([u for u, _ in pairs], dtype=np.int64)
    dst = np.array([v for _, v in pairs], dtype=np.int64)
    times = list(rng.integers(0, 100, size=len(pairs)))
    g = nx.MultiDiGraph()
    g.add_nodes_from((n, {"node_type": "file", "label": n}) for n in nodes)
    g.add_edges_from((nodes[u], nodes[v], {"time": t}) for u, v, t in zip(src, dst, times))
    node_cols = encode_attr_dicts(list(g._node.values()))
    expected = _payload_bytes(encode_graph(g))
    assert _payload_bytes(encode_edge_positions(nodes, node_cols, src, dst, {"time": times})) == expected
    # An int64 array stands for the list of its numpy scalars
    times = np.array(times, dtype=np.int64)
    assert _payload_bytes(encode_edge_positions(nodes, node_cols, src, dst, {"time": times})) == expected
