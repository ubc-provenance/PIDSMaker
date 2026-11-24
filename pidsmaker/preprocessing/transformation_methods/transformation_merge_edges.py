"""Merges the repeated edges of each graph, as done by network IDSs (Euler, Argus, LARES).

All the edges between a same source and destination (and optionally of a same edge type) in a
graph become a single edge, with the attributes of the first event, the label of the most
malicious event (`y` is the max of the merged labels), and two new attributes:

- `count`: the number of merged events;
- `weight`: the count standardized over the edges of the graph and squashed by a sigmoid,
  sigmoid((count - mean) / std), used as edge weight by weighted encoders (e.g. `gcn`).

With `aggregate_attributes`, merged edges also aggregate the edge attributes of all their events
(see `aggregated_edge_attribute_columns`): the number of events with each value of each
categorical attribute, and the number, mean and std of the values of numeric attributes. These
are the edge features of the models instead of the attributes of the first event.

Merged edges are ordered by the time of their first event.
"""

import networkx as nx
import numpy as np
import pandas as pd

from pidsmaker.utils.dataset_utils import aggregated_edge_attribute_columns, get_input_edge_attributes
from pidsmaker.utils.graph_store import graph_columns


def main(G: nx.MultiDiGraph, cfg) -> nx.MultiDiGraph:
    """Merges the edges of a networkx graph (see `merge_columns`)."""
    return merge_columns(graph_columns(G), cfg)


def _factorize(cols_attrs, name, m):
    """(codes, uniques) of an edge attribute column (all None if the column is missing)."""
    if name in cols_attrs:
        return cols_attrs.factorize(name)
    return np.zeros(m, dtype=np.int64), [None]


def merge_columns(cols, cfg) -> nx.MultiDiGraph:
    """Merged graph of the graph whose columns are `cols` (`graph_store.GraphColumns`, e.g. from
    `load_graph_columns`), computed with vectorized operations on the columns: graphs of network
    datasets have millions of edges."""
    per_edge_type = bool(cfg.transformation.merge_edges.per_edge_type)
    aggregate = bool(getattr(cfg.transformation.merge_edges, "aggregate_attributes", False))
    attributes = get_input_edge_attributes(cfg, selected=False) if aggregate else []
    columns = aggregated_edge_attribute_columns(cfg, selected=False) if aggregate else {}

    H = nx.MultiDiGraph()
    node_keys = list(cols.node_attrs)
    H.add_nodes_from(
        (n, dict(zip(node_keys, values)))
        for n, *values in zip(cols.node_ids, *(cols.node_attrs[k] for k in node_keys))
    )
    m = len(cols.src)
    if m == 0:
        return H

    attrs = cols.edge_attrs
    src, dst = np.asarray(cols.src, dtype=np.int64), np.asarray(cols.dst, dtype=np.int64)
    # Groups numbered in order of first appearance, as the dict of `main`
    gid = pd.factorize(src * len(cols.node_ids) + dst)[0]
    if per_edge_type:
        label_codes, labels = _factorize(attrs, "label", m)
        gid = pd.factorize(gid.astype(np.int64) * len(labels) + label_codes)[0]
    gid = gid.astype(np.int64)
    num_groups = int(gid.max()) + 1
    time = attrs.array("time", np.int64)

    # First event of each group: earliest time, then earliest in edge order (stable sort)
    order = np.lexsort((time, gid))
    sorted_gid = gid[order]
    starts = np.flatnonzero(np.r_[True, sorted_gid[1:] != sorted_gid[:-1]])
    first_idx = order[starts]  # indexed by group

    counts = np.bincount(gid, minlength=num_groups)
    if "y" in attrs:
        y_max = np.maximum.reduceat(attrs.array("y", np.int64)[order], starts)
    else:
        y_max = np.zeros(num_groups, dtype=np.int64)

    extra = {}  # aggregated column -> per-group values
    for attr in attributes:
        codes, uniques = _factorize(attrs, attr.name, m)
        if attr.kind == "categorical":
            known = set(attr.values)
            index = {value: i for i, (_, value) in enumerate(columns[attr.name])}
            lut = np.array(
                [index.get(v if v in known else attr.OTHER, -1) for v in map(str, uniques)],
                dtype=np.int64,
            )
            value_idx = lut[codes]
            kept = value_idx >= 0
            k = len(index)
            table = np.bincount(
                gid[kept] * k + value_idx[kept], minlength=num_groups * k
            ).reshape(num_groups, k)
            for i, (column, _) in enumerate(columns[attr.name]):
                extra[column] = table[:, i]
        else:
            lut = np.array([np.nan if v is None else float(v) for v in uniques], dtype=np.float64)
            xs = lut[codes]
            valid = ~np.isnan(xs)
            x = np.where(valid, xs, 0.0)
            n = np.bincount(gid, weights=valid, minlength=num_groups)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, np.bincount(gid, weights=x, minlength=num_groups) / n, 0.0)
                sq = np.where(valid, (x - mean[gid]) ** 2, 0.0)
                var = np.bincount(gid, weights=sq, minlength=num_groups) / n
                std = np.where(n > 0, np.sqrt(var), 0.0)
            stats = {"n": n.astype(np.int64), "mean": mean, "std": std}
            for column, stat in columns[attr.name]:
                extra[column] = stats[stat]

    c = counts.astype(np.float64)
    mean, std = c.mean(), c.std()
    z = (c - mean) / std if std > 0 else np.zeros(num_groups)
    weight = 1.0 / (1.0 + np.exp(-z))

    # Merged edges ordered by the time of their first event (ties: group order)
    group_order = np.lexsort((np.arange(num_groups), time[first_idx]))
    edge_idx = first_idx[group_order]
    keys = list(attrs) + [k for k in ("y", "count") if k not in attrs] + list(extra) + ["weight"]
    values = {k: attrs.take(k, edge_idx) for k in attrs}
    values["y"] = y_max[group_order].tolist()
    values["count"] = counts[group_order].tolist()
    for column, col_values in extra.items():
        values[column] = col_values[group_order].tolist()
    values["weight"] = weight[group_order].tolist()

    node_ids = cols.node_ids
    us = [node_ids[i] for i in src[edge_idx].tolist()]
    vs = [node_ids[i] for i in dst[edge_idx].tolist()]
    H.add_edges_from(
        (u, v, dict(zip(keys, row))) for u, v, *row in zip(us, vs, *(values[k] for k in keys))
    )
    return H
