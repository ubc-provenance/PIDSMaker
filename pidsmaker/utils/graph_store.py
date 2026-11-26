"""Compact on-disk storage of the networkx graphs built by construction/transformation.

Graphs used to be saved as pickled networkx objects, which store every node/edge as
Python dicts of Python objects (e.g. each edge holds a 36-char UUID string, a label
string and two ints). They are now saved as typed column arrays:

- string columns are dictionary-encoded when values repeat (node types, labels,
  edge operations), packed as 16-byte UUIDs when they are canonical UUIDs, stored as
  integers when they are canonical decimal integers (node IDs), or as one UTF-8 blob;
- int/float columns as int64/float64 arrays (np.int64 scalars too, e.g. the random timestamps
  of R-Caid, loaded back as np.int64), anything else as a pickled list;
- the adjacency order (successors and predecessors) is recorded so that the rebuilt
  graph is identical to the original one: same nodes, edges, keys, attributes, Python
  types and iteration orders (nodes, edges, in_edges, predecessors).

The file stays a `torch.save` file whose unpickling rebuilds the networkx graph, so
`torch.load(path)` still returns the graph and every reader works unchanged. Old
pickled graphs remain readable. Unsupported graphs (undirected, subclasses,
heterogeneous attribute keys) are saved as plain pickles, as before.

`load_graph_columns(path)` returns the decoded columns without building networkx
objects, for readers that only need arrays.
"""

import gc
import pickle
import threading
import uuid as _uuid
import zlib
from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd
import torch

FORMAT_VERSION = 1

_SUPPORTED_CLASSES = {nx.MultiDiGraph: "MultiDiGraph", nx.DiGraph: "DiGraph"}
_CLASS_BY_NAME = {v: k for k, v in _SUPPORTED_CLASSES.items()}

_I32 = np.iinfo(np.int32)


# --------------------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------------------


def _pack_array(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, zlib.compress(a.tobytes(), 1))


def _unpack_array(packed):
    dtype, shape, data = packed
    return np.frombuffer(zlib.decompress(data), dtype=np.dtype(dtype)).reshape(shape)


def _smallest_index_dtype(n):
    if n < 2**7:
        return np.int8
    if n < 2**15:
        return np.int16
    return np.int32


# --------------------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------------------


def _all_type(values, t):
    return all(type(v) is t for v in values)


def _try_uuid(values):
    """Packs canonical UUID strings (all upper or all lower case) as 16 bytes each."""
    if not values or not all(len(v) == 36 for v in values):
        return None
    upper = values[0].upper() == values[0]
    try:
        raw = [_uuid.UUID(v).bytes for v in values]
    except ValueError:
        return None
    blob = b"".join(raw)
    if _decode_uuids(np.frombuffer(blob, dtype=np.uint8), len(values), upper) != values:
        return None
    return ("uuid", _pack_array(np.frombuffer(blob, dtype=np.uint8)), upper)


def _decode_uuids(arr, n, upper):
    h = arr.tobytes().hex()
    if upper:
        h = h.upper()
    out = []
    for i in range(n):
        s = h[32 * i : 32 * i + 32]
        out.append(f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}")
    return out


def _try_int_strings(values):
    try:
        ints = [int(v) for v in values]
    except ValueError:
        return None
    if any(str(i) != v for i, v in zip(ints, values)):
        return None
    try:
        return ("intstr", _pack_array(np.array(ints, dtype=np.int64)))
    except OverflowError:
        return None


def _encode_column(values):
    # Types of the values in one pass (an empty column is an int column)
    types = set(map(type, values)) if len(values) else {int}
    if types == {int}:
        try:
            return ("int", _pack_array(np.array(values, dtype=np.int64)))
        except OverflowError:
            return ("obj", values)
    if types == {float}:
        return ("float", _pack_array(np.array(values, dtype=np.float64)))
    if types == {np.int64}:
        # numpy scalars (e.g. drawn by np.random), decoded as numpy scalars
        return ("npint64", _pack_array(np.array(values, dtype=np.int64)))
    if types == {str}:
        if len(set(values)) <= max(1, len(values) // 2):
            # Values numbered in order of first appearance (same as a dict of the values)
            codes, vocab = pd.factorize(np.array(values, dtype=object))
            codes = codes.astype(_smallest_index_dtype(len(vocab)))
            return ("strdict", vocab.tolist(), _pack_array(codes))
        for attempt in (_try_uuid, _try_int_strings):
            col = attempt(values)
            if col is not None:
                return col
        blobs = [v.encode("utf-8", "surrogatepass") for v in values]
        offsets = np.cumsum([0] + [len(b) for b in blobs], dtype=np.int64)
        return ("strblob", zlib.compress(b"".join(blobs), 1), _pack_array(offsets))
    return ("obj", values)


def _decode_column(col, n):
    kind = col[0]
    if kind in ("int", "float"):
        return _unpack_array(col[1]).tolist()
    if kind == "npint64":
        return list(_unpack_array(col[1]))
    if kind == "strdict":
        vocab = col[1]
        return [vocab[i] for i in _unpack_array(col[2]).tolist()]
    if kind == "uuid":
        return _decode_uuids(_unpack_array(col[1]), n, col[2])
    if kind == "intstr":
        return [str(i) for i in _unpack_array(col[1]).tolist()]
    if kind == "strblob":
        blob = zlib.decompress(col[1])
        offsets = _unpack_array(col[2]).tolist()
        return [
            blob[offsets[i] : offsets[i + 1]].decode("utf-8", "surrogatepass") for i in range(n)
        ]
    if kind == "obj":
        return list(col[1])
    raise ValueError(f"Unknown column kind {kind}")


def _encode_attr_dicts(dicts):
    """Encodes a list of attribute dicts sharing the same keys (in the same order)."""
    if not dicts:
        return (), []
    keys = tuple(dicts[0])
    for d in dicts:
        if len(d) != len(keys) or tuple(d) != keys:
            return None
    return keys, [_encode_column([d[k] for d in dicts]) for k in keys]


def encode_attr_dicts(dicts):
    """(keys, encoded columns) of a list of attribute dicts, or None if they do not all have the
    same keys in the same order (such graphs are saved as pickles)."""
    return _encode_attr_dicts(dicts)


def _decode_attr_dicts(keys, cols, n):
    if not keys:
        return [{} for _ in range(n)]
    values = [_decode_column(c, n) for c in cols]
    return [dict(zip(keys, row)) for row in zip(*values)]


# --------------------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------------------


def encode_graph(graph):
    """Returns the compact payload of a graph, or None if it is not supported."""
    cls_name = _SUPPORTED_CLASSES.get(type(graph))
    if cls_name is None:
        return None
    multi = graph.is_multigraph()

    nodes = list(graph._node)
    if list(graph._succ) != nodes or list(graph._pred) != nodes:
        return None
    node_pos = {n: i for i, n in enumerate(nodes)}

    node_attrs = _encode_attr_dicts(list(graph._node.values()))
    if node_attrs is None:
        return None

    # Edges in successor order: for u, for v in succ[u], for key in succ[u][v]
    src, dst, keys, edge_dicts = [], [], [], []
    pair_pos = {}
    for u, nbrs in graph._succ.items():
        ui = node_pos[u]
        for v, inner in nbrs.items():
            vi = node_pos[v]
            pair_pos[(u, v)] = len(pair_pos)
            # The predecessor adjacency must share the successor inner dicts
            if graph._pred[v].get(u) is not inner:
                return None
            if multi:
                if len(inner) == 0:
                    return None
                for k, d in inner.items():
                    src.append(ui)
                    dst.append(vi)
                    keys.append(k)
                    edge_dicts.append(d)
            else:
                src.append(ui)
                dst.append(vi)
                edge_dicts.append(inner)

    # Order of each node's predecessors, as indices of (u, v) pairs in successor order
    pred_order = []
    for v, preds in graph._pred.items():
        for u in preds:
            pred_order.append(pair_pos[(u, v)])
    if len(pred_order) != len(pair_pos):
        return None

    edge_attrs = _encode_attr_dicts(edge_dicts)
    if edge_attrs is None:
        return None

    return _payload(
        cls_name, dict(graph.graph), nodes, node_attrs, src, dst,
        _encode_column(keys) if multi else None, edge_attrs, len(pair_pos), pred_order,
    )


def _payload(cls_name, graph_attrs, nodes, node_attrs, src, dst, keys, edge_attrs, num_pairs, pred_order):
    """Payload of a graph: `node_attrs` and `edge_attrs` are (keys, encoded columns), `src`
    and `dst` the node positions of the edges in successor order, `keys` the encoded edge keys
    (None for simple graphs) and `pred_order` the predecessor order of the (u, v) pairs."""
    idx_dtype = np.int32 if len(nodes) <= _I32.max else np.int64
    return {
        "version": FORMAT_VERSION,
        "class": cls_name,
        "graph_attrs": graph_attrs,
        "num_nodes": len(nodes),
        "num_edges": len(src),
        "num_pairs": num_pairs,
        "node_ids": _encode_column(list(nodes)),
        "node_attr_keys": node_attrs[0],
        "node_attr_cols": node_attrs[1],
        "src": _pack_array(np.asarray(src).astype(idx_dtype)),
        "dst": _pack_array(np.asarray(dst).astype(idx_dtype)),
        "keys": keys,
        "edge_attr_keys": edge_attrs[0],
        "edge_attr_cols": edge_attrs[1],
        "pred_order": _pack_array(np.asarray(pred_order, dtype=np.int64)),
    }


def encode_edge_list(nodes, node_attrs, src, dst, edge_attrs):
    """Payload of the MultiDiGraph built by adding the nodes `nodes` (in this order, with the
    attributes {key: list of values} `node_attrs`) and then the edges (src[i], dst[i]) with the
    attributes {key: list of values} `edge_attrs`, in this order.

    Same result as `encode_graph` on that graph, without building it: networkx graphs store
    each edge as a dict, which is slow to build and to encode for millions of edges.
    """
    num_nodes, num_edges = len(nodes), len(src)
    node_pos = {n: i for i, n in enumerate(nodes)}
    if len(node_pos) != num_nodes:
        raise ValueError("Duplicate nodes")

    src_pos = np.fromiter((node_pos[u] for u in src), dtype=np.int64, count=num_edges)
    dst_pos = np.fromiter((node_pos[v] for v in dst), dtype=np.int64, count=num_edges)
    node_cols = (tuple(node_attrs), [_encode_column(list(v)) for v in node_attrs.values()]) if num_nodes else ((), [])
    return encode_multi_edge_positions(nodes, node_cols, src_pos, dst_pos, edge_attrs)


def encode_multi_edge_positions(nodes, node_cols, src_pos, dst_pos, edge_attrs):
    """Same as `encode_edge_list`, with the edges given as node positions, the node attributes as
    encoded columns (as returned by `encode_attr_dicts`) and the edge attributes as lists or int64
    arrays (whose values are stored as Python ints)."""
    num_nodes, num_edges = len(nodes), len(src_pos)
    src_pos = np.asarray(src_pos, dtype=np.int64)
    dst_pos = np.asarray(dst_pos, dtype=np.int64)
    # Pairs (u, v) numbered in order of first appearance: the order of the successors of u
    # (and of the predecessors of v) in the networkx adjacency dicts
    pair_id, _ = pd.factorize(src_pos * max(num_nodes, 1) + dst_pos)
    pair_id = pair_id.astype(np.int64)
    num_pairs = int(pair_id.max()) + 1 if num_edges else 0

    # Edges in successor order: for u, for v in succ[u], for key in succ[u][v]
    order = np.lexsort((np.arange(num_edges), pair_id, src_pos))
    sorted_pairs = pair_id[order]
    starts = np.r_[True, sorted_pairs[1:] != sorted_pairs[:-1]] if num_edges else np.zeros(0, bool)
    start_idx = np.flatnonzero(starts)
    keys = np.arange(num_edges) - np.repeat(start_idx, np.diff(np.append(start_idx, num_edges)))

    # Index of each pair in successor order, and pairs in predecessor order
    pair_src = np.empty(num_pairs, dtype=np.int64)
    pair_dst = np.empty(num_pairs, dtype=np.int64)
    pair_src[pair_id], pair_dst[pair_id] = src_pos, dst_pos
    succ_pairs = np.lexsort((np.arange(num_pairs), pair_src))
    pair_rank = np.empty(num_pairs, dtype=np.int64)
    pair_rank[succ_pairs] = np.arange(num_pairs)
    pred_pairs = np.lexsort((np.arange(num_pairs), pair_dst))

    if num_edges:
        take = order.tolist()
        edge_cols = (
            tuple(edge_attrs),
            [
                _encode_values(values[order], scalars=False)
                if isinstance(values, np.ndarray)
                else _encode_column([values[i] for i in take])
                for values in edge_attrs.values()
            ],
        )
    else:
        edge_cols = ((), [])
    return _payload(
        "MultiDiGraph", {}, nodes, node_cols, src_pos[order], dst_pos[order],
        _encode_values(keys.astype(np.int64), scalars=False), edge_cols, num_pairs,
        pair_rank[pred_pairs],
    )


def encode_edge_positions(nodes, node_cols, src_pos, dst_pos, edge_attrs):
    """Payload of the MultiDiGraph built by adding the nodes `nodes` (in this order, with the
    attribute columns `node_cols`, as returned by `encode_attr_dicts`) and then the edges
    (nodes[src_pos[i]], nodes[dst_pos[i]]) with the attributes {key: list of values, or int64
    array standing for its list of numpy scalars} `edge_attrs`, in this order, given that each (src, dst) pair appears once and that `src_pos`
    is sorted (edges already in successor order, all with key 0)."""
    src_pos = np.asarray(src_pos, dtype=np.int64)
    dst_pos = np.asarray(dst_pos, dtype=np.int64)
    num_edges = len(src_pos)
    if num_edges and np.any(src_pos[1:] < src_pos[:-1]):
        raise ValueError("Edges are not sorted by source node")
    if num_edges and len(np.unique(src_pos * max(len(nodes), 1) + dst_pos)) != num_edges:
        raise ValueError("Duplicate edges")
    # Predecessors of each node in the order their edges were added
    pred_order = np.argsort(dst_pos, kind="stable")
    edge_cols = (
        (tuple(edge_attrs), [_encode_values(v) for v in edge_attrs.values()])
        if num_edges
        else ((), [])
    )
    keys = _encode_values(np.zeros(num_edges, dtype=np.int64), scalars=False)
    return _payload(
        "MultiDiGraph", {}, nodes, node_cols, src_pos, dst_pos, keys, edge_cols, num_edges,
        pred_order,
    )


def _encode_values(values, scalars=True):
    """`_encode_column` of a list, or of the elements of an int64 array without building them:
    numpy scalars (`list(values)`) or Python ints (`values.tolist()`) if not `scalars`."""
    if isinstance(values, np.ndarray) and values.dtype == np.int64 and values.ndim == 1 and len(values):
        return ("npint64" if scalars else "int", _pack_array(values))
    return _encode_column(list(values) if scalars else values.tolist())


def save_edge_positions(path, nodes, node_cols, src_pos, dst_pos, edge_attrs):
    """Saves the graph described by `encode_edge_positions`, as `save_graph` would save it."""
    obj = _CompactGraph(encode_edge_positions(nodes, node_cols, src_pos, dst_pos, edge_attrs))
    torch.save(obj, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def save_multi_edge_positions(path, nodes, node_cols, src_pos, dst_pos, edge_attrs):
    """Saves the graph described by `encode_multi_edge_positions`, as `save_graph` would save it."""
    obj = _CompactGraph(encode_multi_edge_positions(nodes, node_cols, src_pos, dst_pos, edge_attrs))
    torch.save(obj, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def save_edge_list(path, nodes, node_attrs, src, dst, edge_attrs):
    """Saves the graph described by `encode_edge_list`, as `save_graph` would save it."""
    obj = _CompactGraph(encode_edge_list(nodes, node_attrs, src, dst, edge_attrs))
    torch.save(obj, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def decode_graph(payload):
    """Rebuilds the networkx graph of a payload, identical to the encoded one."""
    # Building many containers triggers costly cyclic GC passes that cannot free anything
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        return _decode_graph(payload)
    finally:
        if gc_enabled:
            gc.enable()


def _decode_graph(payload):
    if payload["version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported graph format version {payload['version']}")
    graph = _CLASS_BY_NAME[payload["class"]]()
    multi = graph.is_multigraph()
    graph.graph.update(payload["graph_attrs"])

    n, m = payload["num_nodes"], payload["num_edges"]
    nodes = _decode_column(payload["node_ids"], n)
    node_dicts = _decode_attr_dicts(payload["node_attr_keys"], payload["node_attr_cols"], n)
    node_store, succ, pred = graph._node, graph._succ, graph._pred
    for node, d in zip(nodes, node_dicts):
        node_store[node] = d
        succ[node] = {}
        pred[node] = {}

    src = _unpack_array(payload["src"])
    dst = _unpack_array(payload["dst"])
    edge_dicts = _decode_attr_dicts(payload["edge_attr_keys"], payload["edge_attr_cols"], m)

    # Edges of a (u, v) pair are contiguous: find where each pair starts
    if m > 0:
        starts = np.flatnonzero(
            np.concatenate(([True], (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])))
        )
    else:
        starts = np.zeros(0, dtype=np.int64)
    node_arr = np.empty(n, dtype=object)
    node_arr[:] = nodes
    pair_u = node_arr[src[starts]].tolist()
    pair_v = node_arr[dst[starts]].tolist()

    if multi:
        keys = _decode_column(payload["keys"], m)
        counts = np.diff(np.append(starts, m)).tolist()
        inners = []
        e = 0
        for u, v, c in zip(pair_u, pair_v, counts):
            if c == 1:
                inner = {keys[e]: edge_dicts[e]}
            else:
                inner = dict(zip(keys[e : e + c], edge_dicts[e : e + c]))
            e += c
            succ[u][v] = inner
            inners.append(inner)
    else:
        inners = edge_dicts
        for u, v, d in zip(pair_u, pair_v, edge_dicts):
            succ[u][v] = d

    for p in _unpack_array(payload["pred_order"]).tolist():
        pred[pair_v[p]][pair_u[p]] = inners[p]
    return graph


def _take_column(col, n, idx):
    """Values of a column at the positions `idx` (same as `_decode_column(col, n)[i]`)."""
    kind = col[0]
    idx = np.asarray(idx, dtype=np.int64)
    if kind in ("int", "float"):
        return _unpack_array(col[1])[idx].tolist()
    if kind == "npint64":
        return list(_unpack_array(col[1])[idx])
    if kind == "strdict":
        vocab = col[1]
        return [vocab[i] for i in _unpack_array(col[2])[idx].tolist()]
    if kind == "uuid":
        arr = _unpack_array(col[1]).reshape(n, 16)[idx]
        return _decode_uuids(arr, len(idx), col[2])
    if kind == "intstr":
        return [str(i) for i in _unpack_array(col[1])[idx].tolist()]
    if kind == "strblob":
        blob = zlib.decompress(col[1])
        offsets = _unpack_array(col[2])
        starts, ends = offsets[idx].tolist(), offsets[idx + 1].tolist()
        return [blob[s:e].decode("utf-8", "surrogatepass") for s, e in zip(starts, ends)]
    if kind == "obj":
        values = col[1]
        return [values[i] for i in idx.tolist()]
    raise ValueError(f"Unknown column kind {kind}")


def _factorize_column(col, n):
    """(codes, uniques) of a column: `uniques[codes[i]]` is its i-th value."""
    kind = col[0]
    if kind == "strdict":
        return _unpack_array(col[2]).astype(np.int64), list(col[1])
    if kind in ("int", "float", "intstr", "npint64"):
        codes, uniques = pd.factorize(_unpack_array(col[1]), use_na_sentinel=False)
        uniques = list(uniques) if kind == "npint64" else uniques.tolist()
        if kind == "intstr":
            uniques = [str(u) for u in uniques]
        return codes.astype(np.int64), uniques
    values = _decode_column(col, n)
    try:
        # Exact values (pandas would turn None into NaN)
        index = {}
        codes = np.fromiter((index.setdefault(v, len(index)) for v in values), np.int64, count=n)
        return codes, list(index)
    except TypeError:  # unhashable values
        codes, uniques = pd.factorize(pd.Series(values, dtype=object), use_na_sentinel=False)
        return codes.astype(np.int64), list(uniques)


class LazyColumns(Mapping):
    """{name: column values (list)}, each column being decoded on first access."""

    def __init__(self, keys, cols, n):
        self._cols = dict(zip(keys, cols))
        self._n = n
        self._decoded = {}

    def __getitem__(self, key):
        if key not in self._decoded:
            self._decoded[key] = _decode_column(self._cols[key], self._n)
        return self._decoded[key]

    def __iter__(self):
        return iter(self._cols)

    def __len__(self):
        return len(self._cols)

    def __contains__(self, key):
        return key in self._cols

    def take(self, key, idx):
        """Values of a column at the positions `idx`, without decoding the whole column."""
        if key in self._decoded:
            values = self._decoded[key]
            return [values[i] for i in np.asarray(idx).tolist()]
        return _take_column(self._cols[key], self._n, idx)

    def encoded(self):
        """(keys, encoded columns), as stored."""
        return tuple(self._cols), list(self._cols.values())

    def factorize(self, key):
        """(codes array, unique values) of a column, without building one object per value
        when the column is stored as integers or a dictionary."""
        return _factorize_column(self._cols[key], self._n)

    def array(self, key, dtype):
        """A numeric column as a numpy array."""
        col = self._cols[key]
        if col[0] in ("int", "float", "npint64"):
            return _unpack_array(col[1]).astype(dtype, copy=False)
        return np.asarray(self[key], dtype=dtype)


class GraphColumns:
    """Columns of a stored graph (see `load_graph_columns`). Node and edge attributes are
    `LazyColumns`, decoded on first access."""

    def __init__(self, payload):
        n, m = payload["num_nodes"], payload["num_edges"]
        self.multi = payload["class"] == "MultiDiGraph"
        self.graph_attrs = payload["graph_attrs"]
        self.node_ids = _decode_column(payload["node_ids"], n)
        self.node_attrs = LazyColumns(payload["node_attr_keys"], payload["node_attr_cols"], n)
        # Edges in `graph.edges` order, as node positions in `node_ids`
        self.src = _unpack_array(payload["src"])
        self.dst = _unpack_array(payload["dst"])
        self.edge_attrs = LazyColumns(payload["edge_attr_keys"], payload["edge_attr_cols"], m)


_LOAD_STATE = threading.local()


def _rebuild(payload):
    if getattr(_LOAD_STATE, "columns", False):
        return GraphColumns(payload)
    return decode_graph(payload)


class _CompactGraph:
    """Pickled in place of the graph: unpickling it calls `_rebuild`."""

    def __init__(self, payload):
        self.payload = payload

    def __reduce__(self):
        return (_rebuild, (self.payload,))


def save_graph(graph, path):
    """Saves a networkx graph compactly (or as a plain pickle if not supported)."""
    payload = encode_graph(graph)
    obj = graph if payload is None else _CompactGraph(payload)
    torch.save(obj, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path):
    """Loads a graph saved by `save_graph` or a legacy pickled graph."""
    return torch.load(path)


def graph_columns(graph):
    """`GraphColumns` of a networkx graph (as if it was saved and loaded with
    `load_graph_columns`). Graphs that cannot be stored compactly (e.g. edges with different
    attributes) are not supported."""
    payload = encode_graph(graph)
    if payload is None:
        raise ValueError(
            "Unsupported graph: only directed (multi)graphs whose nodes and edges all have the "
            "same attributes are supported"
        )
    return GraphColumns(payload)


def load_graph_columns(path):
    """Loads a stored graph as `GraphColumns`, without building networkx objects.

    Returns None for graphs stored as plain pickles (legacy or unsupported), in which
    case `load_graph` should be used.
    """
    _LOAD_STATE.columns = True
    try:
        obj = torch.load(path)
    finally:
        _LOAD_STATE.columns = False
    return obj if isinstance(obj, GraphColumns) else None
