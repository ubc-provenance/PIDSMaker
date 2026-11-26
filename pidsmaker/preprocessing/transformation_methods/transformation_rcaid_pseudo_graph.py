"""R-Caid pseudo-graph transformation.

Each root node (whose earliest outgoing edge precedes its earliest incoming edge) gets an edge
to each of its descendants. With pruning, roots reaching more than half of the nodes get none.
Only these root -> descendant edges are kept, with random timestamps drawn in the range of the
original timestamps.

The options of `transformation.rcaid_pseudo_graph` (see `Options`) follow the paper (Section
4.2): roots are files and sockets (`root_node_types`), and the GNN passes messages over the
provenance graph and over the pseudo edges, whose sources keep their initial features
(`keep_original_edges`: the original edges are kept, and the pseudo edges have no edge type).
They also bound the pseudo-graph, which the paper only prunes with `prune_threshold`: a process
reading thousands of files gives as many roots to each of its descendants (902M pseudo edges in
a window of THEIA_E5 with 146K nodes). Roots with the same label bring the same features and are
merged (`merge_same_label_roots`), and each node keeps its `max_roots_per_node` most specific
roots, those with the fewest descendants (as the paper prunes the roots reaching many nodes).
Unset, the options give the original transformation described below.

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

    def __init__(self, nodes, succ, src, dst, times, node_attr=None):
        self.nodes, self.succ, self.src, self.dst, self.times = nodes, succ, src, dst, times
        self.pos = {n: i for i, n in enumerate(nodes)}
        # node_attr(key): the values of a node attribute, in node order
        self.node_attr = node_attr

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
        node_store = G._node
        return cls(
            nodes, succ, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), times,
            node_attr=lambda key: [node_store[n].get(key) for n in nodes],
        )

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
        return cls(nodes, succ, src, dst, times, node_attr=lambda key: cols.node_attrs[key])


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

    def __init__(self, structure, use_pruning, prune_threshold=PRUNE_THRESHOLD):
        self.nodes = structure.nodes
        self.pos = structure.pos
        self.succ = structure.succ
        n = len(self.nodes)
        self.pred = None  # built on first use
        # Original rule: pruned if the number of descendants is > prune_threshold * |V|
        self.max_desc = prune_threshold * n if use_pruning else None
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




class Options:
    """Options of the transformation (`transformation.rcaid_pseudo_graph`). Unset options keep
    the behavior of the original implementation."""

    def __init__(self, cfg):
        c = cfg.transformation.rcaid_pseudo_graph
        self.use_pruning = bool(c.use_pruning)
        threshold = getattr(c, "prune_threshold", None)
        self.prune_threshold = PRUNE_THRESHOLD if threshold is None else float(threshold)
        types = (getattr(c, "root_node_types", None) or "all").strip()
        self.root_node_types = (
            None if types == "all" else {t.strip() for t in types.split(",") if t.strip()}
        )
        self.keep_original_edges = bool(getattr(c, "keep_original_edges", None) or False)
        self.merge_same_label_roots = bool(getattr(c, "merge_same_label_roots", None) or False)
        max_roots = getattr(c, "max_roots_per_node", None)
        self.max_roots_per_node = int(max_roots) if max_roots else None  # 0 or unset: no limit


def _factorize(values):
    """Integer codes of hashable values (equal values have equal codes)."""
    index = {}
    return np.fromiter((index.setdefault(v, len(index)) for v in values), np.int64, count=len(values))


def pseudo_graph_edges(structure, use_pruning, options=None):
    """Pseudo edges (root -> descendant) of the transformed graph, in order, as arrays of node
    positions (src, dst).

    Without options, as in the original implementation: every root gets an edge to each of its
    descendants (in the order of `nx.descendants`), roots in node order. With options:

    - `root_node_types`: only the roots of these types get edges;
    - `merge_same_label_roots`: a node gets a single pseudo edge from all its roots with the same
      (node type, label), which bring the same features;
    - `max_roots_per_node`: a node gets pseudo edges from at most this number of roots, the most
      specific ones: those with the fewest descendants (then in node order). This bounds the
      number of pseudo edges by `max_roots_per_node` times the number of nodes.

    Edges are ordered by root (in node order), then as without options. Merged roots are
    represented by the first of them processed: in node order without `max_roots_per_node`, by
    specificity otherwise.
    """
    st = structure
    prune_threshold = PRUNE_THRESHOLD if options is None else options.prune_threshold
    mask = _root_mask(len(st.nodes), st.src, st.dst, st.times)
    if options is not None and options.root_node_types is not None:
        types = st.node_attr("node_type")
        mask &= np.fromiter((t in options.root_node_types for t in types), bool, count=len(types))
    roots = np.flatnonzero(mask).tolist()
    empty = np.zeros(0, dtype=np.int64)
    if not roots:
        return empty, empty

    merge = options is not None and options.merge_same_label_roots
    if merge:
        key = _factorize(list(zip(st.node_attr("node_type"), st.node_attr("label"))))
    max_roots = None if options is None else options.max_roots_per_node

    desc = _Descendants(st, use_pruning, prune_threshold)
    # Roots with a single successor first: they are the most likely to identify big nodes
    results = {}
    for r in sorted(roots, key=lambda r: len(st.succ[r]) != 1):
        results[r] = desc.of_root(r)
    roots = [r for r in roots if results[r] is not None and len(results[r])]
    if not roots:
        return empty, empty

    if max_roots is not None:
        return _capped_pseudo_edges(len(st.nodes), roots, results, max_roots, key if merge else None)

    out_src, out_dst = [], []
    # With merging, the roots with the same successors have the same descendants: only the first
    # root of each key gets edges there. The (key, descendant) pairs of the roots with different
    # successors are deduplicated afterwards
    seen_keys = {}
    for r in roots:
        d = results[r]
        if merge:
            group = seen_keys.setdefault(tuple(st.succ[r]), set())
            if key[r] in group:
                continue
            group.add(key[r])
        out_src.append(np.full(len(d), r, dtype=np.int64))
        out_dst.append(d)
    if not out_src:
        return empty, empty
    src, dst = np.concatenate(out_src), np.concatenate(out_dst)
    if merge:
        _, first = np.unique(key[src] * len(st.nodes) + dst, return_index=True)
        first.sort()
        src, dst = src[first], dst[first]
    return src, dst


def _capped_pseudo_edges(num_nodes, roots, results, max_roots, key=None):
    """Pseudo edges when each node keeps at most `max_roots` roots, the ones with the fewest
    descendants first (then in node order); with `key`, at most one root per key and node."""
    count = np.zeros(num_nodes, dtype=np.int64)
    # Keys of the roots of each node (-1: none), to merge the roots with the same key
    node_keys = np.full((num_nodes, max_roots), -1, dtype=np.int64) if key is not None else None
    selected = {}
    for r in sorted(roots, key=lambda r: len(results[r])):
        d = results[r]
        d = d[count[d] < max_roots]
        if key is not None and len(d):
            d = d[~(node_keys[d] == key[r]).any(axis=1)]
            node_keys[d, count[d]] = key[r]
        if len(d):
            count[d] += 1
            selected[r] = d
    empty = np.zeros(0, dtype=np.int64)
    if not selected:
        return empty, empty
    order = sorted(selected)
    src = np.concatenate([np.full(len(selected[r]), r, dtype=np.int64) for r in order])
    dst = np.concatenate([selected[r] for r in order])
    return src, dst


def _transform(structure, cfg):
    """(src, dst, times) of the pseudo edges of the transformed graph."""
    options = Options(cfg)
    src, dst = pseudo_graph_edges(structure, options.use_pruning, options)
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


def _pseudo_edge_attr(key, num_pseudo, pseudo_times):
    """Values of the edge attribute `key` for the pseudo edges: a random time, a `y` of 0, and
    None otherwise (in particular no edge type `label`)."""
    if key == "time":
        return [int(t) for t in pseudo_times]
    if key == "y":
        return [0] * num_pseudo
    return [None] * num_pseudo


def _all_edge_attrs(edge_keys, orig_attr, pseudo_times):
    """{key: values} of the original edges (`orig_attr(key)`) followed by the pseudo edges."""
    out = {}
    for k in edge_keys:
        values = list(orig_attr(k))
        if k == "time":
            values = [int(t) for t in values]
        out[k] = values + _pseudo_edge_attr(k, len(pseudo_times), pseudo_times)
    return out


def _with_original_edges(G, nodes, src, dst, rand_t):
    """MultiDiGraph with the nodes and edges of G (copied attributes), followed by the pseudo
    edges."""
    H = nx.MultiDiGraph()
    H.add_nodes_from((n, d.copy()) for n, d in G._node.items())
    edge_dicts = [d for _, _, d in G.edges(data=True)]
    H.add_edges_from((u, v, d.copy()) for u, v, d in G.edges(data=True))
    edge_keys = list(edge_dicts[0]) if edge_dicts else ["time"]
    columns = {k: _pseudo_edge_attr(k, len(src), rand_t) for k in edge_keys}
    for i, (u, v) in enumerate(zip(np.asarray(src).tolist(), np.asarray(dst).tolist())):
        H.add_edge(nodes[u], nodes[v], **{k: columns[k][i] for k in edge_keys})
    return H


def main(graph: nx.Graph, cfg) -> nx.Graph:
    structure = _Structure.from_graph(graph)
    src, dst, rand_t = _transform(structure, cfg)
    if Options(cfg).keep_original_edges:
        return _with_original_edges(graph, structure.nodes, src, dst, rand_t)
    return _build_multidigraph(graph, structure.nodes, src, dst, rand_t)


def transform_and_save(graph, cfg, path):
    """Saves `main(graph, cfg)` to `path` as `save_graph` would, without building it."""
    structure = _Structure.from_graph(graph)
    src, dst, rand_t = _transform(structure, cfg)
    node_cols = graph_store.encode_attr_dicts(list(graph._node.values()))
    keep_original = Options(cfg).keep_original_edges
    if node_cols is None:  # not storable as columns: saved as a pickle
        if keep_original:
            graph = _with_original_edges(graph, structure.nodes, src, dst, rand_t)
        else:
            graph = _build_multidigraph(graph, structure.nodes, src, dst, rand_t)
        graph_store.save_graph(graph, path)
        return
    if keep_original:
        edge_dicts = [d for _, _, d in graph.edges(data=True)]
        edge_keys = list(edge_dicts[0]) if edge_dicts else ["time"]
        attrs = _all_edge_attrs(edge_keys, lambda k: [d.get(k) for d in edge_dicts], rand_t)
        graph_store.save_multi_edge_positions(
            path, structure.nodes, node_cols, np.concatenate([structure.src, src]),
            np.concatenate([structure.dst, dst]), attrs,
        )
        return
    graph_store.save_edge_positions(path, structure.nodes, node_cols, src, dst, {"time": rand_t})


def transform_columns_and_save(cols, cfg, path):
    """Same as `transform_and_save` on the graph stored with the `GraphColumns` cols."""
    structure = _Structure.from_columns(cols)
    src, dst, rand_t = _transform(structure, cfg)
    node_cols = cols.node_attrs.encoded()
    if Options(cfg).keep_original_edges:
        edge_keys = list(cols.edge_attrs) or ["time"]
        attrs = _all_edge_attrs(edge_keys, lambda k: cols.edge_attrs[k], rand_t)
        graph_store.save_multi_edge_positions(
            path, structure.nodes, node_cols, np.concatenate([structure.src, src]),
            np.concatenate([structure.dst, dst]), attrs,
        )
        return
    graph_store.save_edge_positions(path, structure.nodes, node_cols, src, dst, {"time": rand_t})
