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

import math
from collections import Counter, defaultdict

import networkx as nx

from pidsmaker.utils.dataset_utils import aggregated_edge_attribute_columns, get_input_edge_attributes


def _aggregate(events, attributes, columns):
    """{aggregated column: value} of the events of a merged edge."""
    out = {}
    for attr in attributes:
        values = [ev.get(attr.name) for ev in events]
        if attr.kind == "categorical":
            known = set(attr.values)
            counts = Counter(v if v in known else attr.OTHER for v in map(str, values))
            for column, value in columns[attr.name]:
                out[column] = counts.get(value, 0)
        else:
            xs = [float(v) for v in values if v is not None and float(v) == float(v)]
            mean = sum(xs) / len(xs) if xs else 0.0
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) if xs else 0.0
            stats = {"n": len(xs), "mean": mean, "std": std}
            for column, stat in columns[attr.name]:
                out[column] = stats[stat]
    return out


def main(G: nx.MultiDiGraph, cfg) -> nx.MultiDiGraph:
    per_edge_type = bool(cfg.transformation.merge_edges.per_edge_type)
    aggregate = bool(getattr(cfg.transformation.merge_edges, "aggregate_attributes", False))
    # All the attributes are aggregated: those used as features are selected by feat_inference
    attributes = get_input_edge_attributes(cfg, selected=False) if aggregate else []
    columns = aggregated_edge_attribute_columns(cfg, selected=False) if aggregate else {}

    groups = defaultdict(list)
    for u, v, attr in G.edges(data=True):
        key = (u, v, attr.get("label")) if per_edge_type else (u, v)
        groups[key].append(attr)

    merged = []
    for key, events in groups.items():
        events.sort(key=lambda a: a["time"])
        first = dict(events[0])
        first["y"] = max(int(a.get("y", 0)) for a in events)
        first["count"] = len(events)
        if aggregate:
            first.update(_aggregate(events, attributes, columns))
        merged.append((key[0], key[1], first))
    merged.sort(key=lambda e: e[2]["time"])

    counts = [e[2]["count"] for e in merged]
    mean = sum(counts) / len(counts) if counts else 0.0
    std = math.sqrt(sum((c - mean) ** 2 for c in counts) / len(counts)) if counts else 0.0
    for _, _, attr in merged:
        z = (attr["count"] - mean) / std if std > 0 else 0.0
        attr["weight"] = 1.0 / (1.0 + math.exp(-z))

    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(merged)
    return H
