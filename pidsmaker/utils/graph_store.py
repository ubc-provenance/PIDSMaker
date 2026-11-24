"""Compact on-disk storage of the networkx graphs built by construction/transformation.

Graphs used to be saved as pickled networkx objects, which store every node/edge as
Python dicts of Python objects (e.g. each edge holds a 36-char UUID string, a label
string and two ints). They are now saved as typed column arrays:

- string columns are dictionary-encoded when values repeat (node types, labels,
  edge operations), packed as 16-byte UUIDs when they are canonical UUIDs, stored as
  integers when they are canonical decimal integers (node IDs), or as one UTF-8 blob;
- int/float columns as int64/float64 arrays, anything else as a pickled list;
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

import networkx as nx
import numpy as np
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
    if _all_type(values, int):
        try:
            return ("int", _pack_array(np.array(values, dtype=np.int64)))
        except OverflowError:
            return ("obj", values)
    if _all_type(values, float):
        return ("float", _pack_array(np.array(values, dtype=np.float64)))
    if _all_type(values, str):
        vocab = dict.fromkeys(values)
        if len(vocab) <= max(1, len(values) // 2):
            index = {v: i for i, v in enumerate(vocab)}
            codes = np.array([index[v] for v in values], dtype=_smallest_index_dtype(len(vocab)))
            return ("strdict", list(vocab), _pack_array(codes))
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

    idx_dtype = np.int32 if len(nodes) <= _I32.max else np.int64
    return {
        "version": FORMAT_VERSION,
        "class": cls_name,
        "graph_attrs": dict(graph.graph),
        "num_nodes": len(nodes),
        "num_edges": len(edge_dicts),
        "num_pairs": len(pair_pos),
        "node_ids": _encode_column(nodes),
        "node_attr_keys": node_attrs[0],
        "node_attr_cols": node_attrs[1],
        "src": _pack_array(np.array(src, dtype=idx_dtype)),
        "dst": _pack_array(np.array(dst, dtype=idx_dtype)),
        "keys": _encode_column(keys) if multi else None,
        "edge_attr_keys": edge_attrs[0],
        "edge_attr_cols": edge_attrs[1],
        "pred_order": _pack_array(np.array(pred_order, dtype=np.int64)),
    }


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


class GraphColumns:
    """Decoded columns of a stored graph (see `load_graph_columns`)."""

    def __init__(self, payload):
        n, m = payload["num_nodes"], payload["num_edges"]
        self.multi = payload["class"] == "MultiDiGraph"
        self.graph_attrs = payload["graph_attrs"]
        self.node_ids = _decode_column(payload["node_ids"], n)
        self.node_attrs = {
            k: _decode_column(c, n)
            for k, c in zip(payload["node_attr_keys"], payload["node_attr_cols"])
        }
        # Edges in `graph.edges` order, as node positions in `node_ids`
        self.src = _unpack_array(payload["src"])
        self.dst = _unpack_array(payload["dst"])
        self.edge_attrs = {
            k: _decode_column(c, m)
            for k, c in zip(payload["edge_attr_keys"], payload["edge_attr_cols"])
        }


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
