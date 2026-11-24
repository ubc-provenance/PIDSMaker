"""R-Caid pseudo-graph transformation.

Each root node (whose earliest outgoing edge precedes its earliest incoming edge) gets an edge
to each of its descendants. With pruning, roots reaching more than half of the nodes get none.
Only these root -> descendant edges are kept, with random timestamps drawn in the range of the
original timestamps.

The result is identical to the original implementation (kept in
`tests/test_rcaid_transformation.py`), which built a pseudo-graph with one pseudo-root per root
connected to `nx.descendants(G, root)`, pruned it, merged the pseudo-roots into their roots with
`nx.relabel_nodes`, added the timestamps and converted the result to a MultiDiGraph. Its output
is fully determined by:

- the nodes of G, in order, with copies of their attributes;
- for each kept root in node order, its descendants in the iteration order of the set returned
  by `nx.descendants`, i.e. a set built by inserting the descendants in breadth-first order
  (the iteration order of a set depends on its insertion order, so the set is rebuilt the same
  way);
- `np.random.randint(min_t, max_t, size=num_edges)` assigned to the edges in that order.

It is computed here without building the pseudo-graph and without exploring the descendants of
pruned roots:

- roots are found with numpy from the edge timestamps;
- the breadth-first order from a root r only depends on its successors S (the BFS starts with
  r's successors, in adjacency order), and is the same for all roots with the same S, except
  that r itself is skipped when it is reachable from S (r is then in the same strongly
  connected component as one of its successors). Roots with the same successors (e.g. files
  read by the same process) thus share one BFS;
- with pruning, a BFS stops as soon as it has seen enough nodes for the root to be pruned. Nodes
  known to reach too many nodes ("big") are marked with all their ancestors, so that the BFS of
  later roots stop when they meet one.
"""

import gc

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pidsmaker.utils import graph_store

from .transformation_utils import randint_timestamps

PRUNE_THRESHOLD = 0.5


class _Structure:
    """What the transformation reads from a graph: `nodes` (in order), `succ` (the unique
    successors of each node, as positions in `nodes`, in adjacency order: the order in which
    `nx.bfs_edges` visits them), and the ends (node positions) and times of all the edges, in
    `G.edges(data=True)` order."""

    def __init__(self, nodes, succ, src, dst, times):
        self.nodes, self.succ, self.src, self.dst, self.times = nodes, succ, src, dst, times
        self.pos = {n: i for i, n in enumerate(nodes)}

    @classmethod
    def from_graph(cls, G):
        nodes = list(G._node)
        pos = {n: i for i, n in enumerate(nodes)}
        succ = [[pos[v] for v in G._succ[n]] for n in nodes]
        src, dst, times = [], [], []
        multi = G.is_multigraph()
        for u, nbrs in G._succ.items():
            ui = pos[u]
            for v, inner in nbrs.items():
                vi = pos[v]
                if multi:
                    for d in inner.values():
                        src.append(ui)
                        dst.append(vi)
                        times.append(d["time"])
                else:
                    src.append(ui)
                    dst.append(vi)
                    times.append(inner["time"])
        return cls(nodes, succ, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), times)

    @classmethod
    def from_columns(cls, cols):
        """From the `GraphColumns` of a stored graph (edges in `G.edges` order, so grouped by
        source node in node order, and by (source, destination) pair)."""
        nodes = cols.node_ids
        src = cols.src.astype(np.int64)
        dst = cols.dst.astype(np.int64)
        n, m = len(nodes), len(src)
        if m:
            starts = np.flatnonzero(np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])])
        else:
            starts = np.zeros(0, dtype=np.int64)
        pair_src, pair_dst = src[starts], dst[starts]
        bounds = np.searchsorted(pair_src, np.arange(n + 1)).tolist()
        pair_dst = pair_dst.tolist()
        succ = [pair_dst[bounds[i] : bounds[i + 1]] for i in range(n)]
        times = cols.edge_attrs["time"] if m else []
        return cls(nodes, succ, src, dst, times)


def _numeric_times(times):
    """The times as a numpy array if they can be compared exactly with numpy, else None."""
    types = set(map(type, times))
    try:
        if types <= {int, np.int64}:
            return np.array(times, dtype=np.int64)
        if types == {float}:
            arr = np.array(times, dtype=np.float64)
            return None if np.isnan(arr).any() else arr
    except OverflowError:
        pass
    return None


def _root_mask(num_nodes, src, dst, times):
    """Boolean mask of the root nodes (same rule as `identify_root_nodes` of the original)."""
    arr = _numeric_times(times)
    if arr is None:
        # Exact Python comparison of arbitrary values
        out_min, in_min = {}, {}
        for u, v, t in zip(src.tolist(), dst.tolist(), times):
            if u not in out_min or t < out_min[u]:
                out_min[u] = t
            if v not in in_min or t < in_min[v]:
                in_min[v] = t
        mask = np.zeros(num_nodes, dtype=bool)
        for u, t in out_min.items():
            if u not in in_min or t < in_min[u]:
                mask[u] = True
        return mask

    src_a, dst_a = src, dst
    has_out = np.bincount(src_a, minlength=num_nodes) > 0
    has_in = np.bincount(dst_a, minlength=num_nodes) > 0
    out_min = np.zeros(num_nodes, dtype=arr.dtype)
    in_min = np.zeros(num_nodes, dtype=arr.dtype)
    # Min per node: sort by (node, time) and take the first time of each node
    order = np.lexsort((arr, src_a))
    first = np.r_[True, src_a[order][1:] != src_a[order][:-1]] if len(order) else order.astype(bool)
    out_min[src_a[order][first]] = arr[order][first]
    order = np.lexsort((arr, dst_a))
    first = np.r_[True, dst_a[order][1:] != dst_a[order][:-1]] if len(order) else order.astype(bool)
    in_min[dst_a[order][first]] = arr[order][first]
    return has_out & (~has_in | (out_min < in_min))


class _Descendants:
    """Descendants of the roots in the order of the original implementation."""

    def __init__(self, structure, use_pruning):
        self.nodes = structure.nodes
        self.pos = structure.pos
        self.succ = structure.succ
        n = len(self.nodes)
        self.pred = None  # built on first use
        # Original rule: pruned if the number of descendants is > PRUNE_THRESHOLD * |V|
        self.max_desc = PRUNE_THRESHOLD * n if use_pruning else None
        # A BFS from a successor set S stops once |reach(S)| > max_desc + 1: then every root
        # with these successors has > max_desc descendants
        self.big = bytearray(n)  # nodes v with |reach(v) including v| > max_desc + 1
        self.seen = bytearray(n)
        self.cache = {}

        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum([len(s) for s in self.succ], out=indptr[1:])
        indices = np.fromiter((v for s in self.succ for v in s), dtype=np.int64, count=indptr[-1])
        adj = sp.csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr), shape=(n, n))
        _, self.scc = connected_components(adj, directed=True, connection="strong")
        self.scc = self.scc.tolist()

    def _mark_big(self, v):
        """Marks v and all its ancestors as big."""
        if self.big[v]:
            return
        if self.pred is None:
            self.pred = [[] for _ in range(len(self.succ))]
            for u, s in enumerate(self.succ):
                for w in s:
                    self.pred[w].append(u)
        big, pred = self.big, self.pred
        big[v] = 1
        stack = [v]
        while stack:
            for u in pred[stack.pop()]:
                if not big[u]:
                    big[u] = 1
                    stack.append(u)

    def _bfs(self, start):
        """Nodes reachable from `start` (included) in BFS order, or None if too many (or if a
        big node is reachable)."""
        seen, succ, big = self.seen, self.succ, self.big
        limit = None if self.max_desc is None else self.max_desc + 1
        order = list(start)
        for v in order:
            seen[v] = 1
        too_many = False
        i = 0
        while i < len(order):
            for v in succ[order[i]]:
                if not seen[v]:
                    if big[v]:
                        too_many = True
                        break
                    seen[v] = 1
                    order.append(v)
            if too_many or (limit is not None and len(order) > limit):
                too_many = True
                break
            i += 1
        for v in order:
            seen[v] = 0
        return None if too_many else order

    def _set_order(self, order):
        """Positions of the nodes `order` in the iteration order of a set of their IDs filled in
        this order (as `nx.descendants` returns them)."""
        ids = set(map(self.nodes.__getitem__, order))
        return np.fromiter(map(self.pos.__getitem__, ids), dtype=np.int64, count=len(ids))

    def of_root(self, r):
        """Positions of the descendants of root r in output order, or None if pruned."""
        start = tuple(v for v in self.succ[r] if v != r)
        if not start:
            return np.zeros(0, dtype=np.int64)
        scc = self.scc
        in_cycle = any(scc[v] == scc[r] for v in start)
        if not in_cycle and start in self.cache:
            return self.cache[start]

        if self.max_desc is not None and any(self.big[v] for v in start):
            result = None
        else:
            order = self._bfs(start)
            if order is None:
                result = None
                if len(start) == 1:
                    self._mark_big(start[0])
            else:
                if in_cycle:
                    # r is reached again from its successors, but nx.descendants skips it.
                    # As r only leads to nodes already seen, the order is otherwise the same
                    order = [v for v in order if v != r]
                if self.max_desc is not None and len(order) > self.max_desc:
                    result = None
                else:
                    result = self._set_order(order)
        if result is None and self.max_desc is not None:
            # More than max_desc descendants: reach(r) has more than max_desc + 1 nodes
            self._mark_big(r)
        if not in_cycle:
            self.cache[start] = result
        return result


def pseudo_graph_edges(structure, use_pruning):
    """Edges of the transformed graph, in order, as arrays of node positions (src, dst)."""
    st = structure
    roots = np.flatnonzero(_root_mask(len(st.nodes), st.src, st.dst, st.times)).tolist()
    empty = np.zeros(0, dtype=np.int64)
    if not roots:
        return empty, empty

    desc = _Descendants(st, use_pruning)
    # Roots with a single successor first: they are the most likely to identify big nodes
    results = {}
    for r in sorted(roots, key=lambda r: len(st.succ[r]) != 1):
        results[r] = desc.of_root(r)
    out_src, out_dst = [], []
    for r in roots:
        d = results[r]
        if d is not None and len(d):
            out_src.append(np.full(len(d), r, dtype=np.int64))
            out_dst.append(d)
    if not out_src:
        return empty, empty
    return np.concatenate(out_src), np.concatenate(out_dst)


def _transform(structure, cfg):
    """(src, dst, times) of the edges of the transformed graph."""
    use_pruning = cfg.transformation.rcaid_pseudo_graph.use_pruning
    src, dst = pseudo_graph_edges(structure, use_pruning)
    times = structure.times
    # Raises ValueError on graphs without edges, as the original
    min_t, max_t = min(times), max(times)
    return src, dst, randint_timestamps(min_t, max_t, len(src))


def _build_multidigraph(G, nodes, src, dst, edge_times):
    """MultiDiGraph with the nodes of G (copied attributes) and the edges (key 0, {"time": t})
    added in order, as `nx.MultiDiGraph(DiGraph)` builds it."""
    gc_enabled = gc.isenabled()
    gc.disable()  # building many containers triggers costly GC passes that free nothing
    try:
        H = nx.MultiDiGraph()
        node_store, succ, pred = H._node, H._succ, H._pred
        for n, d in G._node.items():
            node_store[n] = d.copy()
            succ[n] = {}
            pred[n] = {}
        node_arr = np.empty(len(nodes), dtype=object)
        node_arr[:] = nodes
        for u, v, t in zip(node_arr[src].tolist(), node_arr[dst].tolist(), edge_times):
            inner = {0: {"time": t}}
            succ[u][v] = inner
            pred[v][u] = inner
    finally:
        if gc_enabled:
            gc.enable()
    return H


def main(graph: nx.Graph, cfg) -> nx.Graph:
    structure = _Structure.from_graph(graph)
    src, dst, rand_t = _transform(structure, cfg)
    return _build_multidigraph(graph, structure.nodes, src, dst, rand_t)


def transform_and_save(graph, cfg, path):
    """Saves `main(graph, cfg)` to `path` as `save_graph` would, without building it."""
    structure = _Structure.from_graph(graph)
    src, dst, rand_t = _transform(structure, cfg)
    node_cols = graph_store.encode_attr_dicts(list(graph._node.values()))
    if node_cols is None:  # not storable as columns: saved as a pickle
        graph_store.save_graph(_build_multidigraph(graph, structure.nodes, src, dst, rand_t), path)
        return
    graph_store.save_edge_positions(path, structure.nodes, node_cols, src, dst, {"time": rand_t})


def transform_columns_and_save(cols, cfg, path):
    """Same as `transform_and_save` on the graph stored with the `GraphColumns` cols."""
    structure = _Structure.from_columns(cols)
    src, dst, rand_t = _transform(structure, cfg)
    node_cols = cols.node_attrs.encoded()
    graph_store.save_edge_positions(path, structure.nodes, node_cols, src, dst, {"time": rand_t})
