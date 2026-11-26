"""Access to the content of datasets (node and edge types, database tables, ground truth helpers).

Datasets are described by their schema (see `pidsmaker/config/schemas.py`): all functions
here read the node types, tables and edge type vocabulary from the schema of the dataset,
so that they work for any dataset.
"""

import os

import torch

from pidsmaker.config import get_node_feats_from_cfg
from pidsmaker.config.schemas import (  # noqa: F401 (re-exported vocabularies)
    NULL_VALUES,
    OPTC_TO_TC_EDGE_MAP,
    OPTC_hostname_map,
    edge_reversed,
    exclude_edge_type,
    get_schema,
    ntype2id,
    possible_events,
    possible_events_darpa_tc,
    possible_events_graph_processor_carbon_black_edr,
    possible_events_optc,
    possible_events_optc_consistent,
    rel2id_atlasv2,
    rel2id_darpa_tc,
    rel2id_graph_processor_carbon_black_edr,
    rel2id_optc,
    rel2id_optc_consistent,
    translate_optc_to_tc,
)
from pidsmaker.utils.utils import init_database_connection


def decrement_dict(d):
    return {
        k - 1 if isinstance(k, int) else k: v - 1 if isinstance(v, int) else v for k, v in d.items()
    }


def get_rel2id(cfg, from_zero=False):
    """Bidirectional {edge type: id, id: edge type} vocabulary of the graphs of the dataset."""
    rel2id = get_schema(cfg).rel2id
    return decrement_dict(rel2id) if from_zero else rel2id


def get_node_map(cfg, from_zero=False):
    """Bidirectional {node type: id, id: node type} mapping of the dataset."""
    node_map = get_schema(cfg).ntype2id
    return decrement_dict(node_map) if from_zero else node_map


def get_num_edge_type(cfg):
    schema = get_schema(cfg)
    if schema.supports_edge_type_triplets and "edge_type_triplet" in cfg.batching.edge_features:
        return sum([len(events) for events in schema.possible_events.values()])
    return schema.num_edge_types


def get_rel2id_considering_triplets(cfg):
    if "edge_type_triplet" in cfg.batching.edge_features:
        possible_events = get_possible_events(cfg)
        return {
            i + 1: e
            for i, e in enumerate(
                [event for events in possible_events.values() for event in events]
            )
        }
    return get_rel2id(cfg)


def get_possible_events(cfg):
    """{(source type, destination type): [edge types]} of the dataset."""
    return get_schema(cfg).possible_events


_cursor_id = 0


def _iter_rows(connect, sql, fetch_size=100_000):
    """Yields the rows of `sql` through a server-side cursor, `fetch_size` rows at a time,
    instead of transferring all of them at once (`fetchall` of the 44.5M nodes of TRACE_E3
    holds tens of GB of row tuples)."""
    global _cursor_id
    _cursor_id += 1
    cur = connect.cursor(f"pids_rows_cur_{_cursor_id}")
    try:
        cur.execute(sql)
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            yield from rows
    finally:
        cur.close()


def iter_nodes(cfg, connect, node_types=None, fetch_size=50_000):
    """Yields (node_type, node_uuid, hash_id, index_id, attrs) for all nodes of the database.

    Tables are read in the order of the schema's node types, each through a server-side cursor
    so that client memory stays bounded. `attrs` maps each attribute of the node type to the
    string of its database value, plus `type` (the node type's name).

    Args:
        cfg: config of the dataset
        connect: database connection
        node_types: names of the node types to read (default: all types of the schema)
    """
    global _cursor_id
    for nt in get_schema(cfg).node_types:
        if node_types is not None and nt.name not in node_types:
            continue
        columns = ", ".join(["node_uuid", "hash_id", "index_id"] + [col for _, col in nt.columns])
        _cursor_id += 1
        cur = connect.cursor(f"pids_nodes_cur_{_cursor_id}")
        try:
            cur.execute(f"SELECT {columns} FROM {nt.table};")
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                for row in rows:
                    attrs = {"type": nt.name}
                    for (attr, _), value in zip(nt.columns, row[3:]):
                        attrs[attr] = str(value)
                    yield nt.name, str(row[0]), str(row[1]), row[2], attrs
        finally:
            cur.close()


# Last result of `get_uuid_to_index_id`: the evaluation calls it many times per run
_UUID_TO_INDEX_ID_CACHE = {}


def get_uuid_to_index_id(cfg):
    """Returns ({node_uuid: index_id}, {index_id: node_uuid}). The dicts are shared: read only."""
    dataset = cfg.dataset.name
    schema = get_schema(cfg)
    database = (
        cfg.dataset.database_all_file if cfg.construction.use_all_files else cfg.dataset.database
    )
    key = (dataset, database, cfg.database.datasets_dir, schema.name)

    if key not in _UUID_TO_INDEX_ID_CACHE:
        uuid2nid = {}
        nid2uuid = {}

        cur, conn = init_database_connection(cfg)
        try:
            for node_type in get_node_feats_from_cfg(cfg):
                table_name = schema.node_type(node_type).table
                sql = f"SELECT index_id, node_uuid FROM {table_name};"
                for row in _iter_rows(conn, sql):
                    node_uuid, index_id = str(row[1]), int(row[0])
                    uuid2nid[node_uuid] = index_id
                    nid2uuid[index_id] = node_uuid
        finally:
            conn.close()
        _UUID_TO_INDEX_ID_CACHE.clear()  # only keep the last database in memory
        _UUID_TO_INDEX_ID_CACHE[key] = (uuid2nid, nid2uuid)

    # Shared dicts (copying millions of entries at each call is slow): do not modify them
    return _UUID_TO_INDEX_ID_CACHE[key]


def get_time_and_end_node(cfg, start_time, end_time):
    """Return events in the window as (src_index_id, dst_index_id, timestamp, operation).

    src/dst are integer graph index_ids, so callers can compare them directly
    against the index_id-based ground truth.
    """
    schema = get_schema(cfg)
    cur, conn = init_database_connection(cfg)
    try:
        sql = (
            f"SELECT src_index_id, dst_index_id, timestamp_rec, operation FROM {schema.event_table} "
            f"WHERE timestamp_rec > {start_time} AND timestamp_rec < {end_time}{schema.operations_sql()};"
        )
        cur.execute(sql)
        data = cur.fetchall()
    finally:
        conn.close()
    return [(int(r[0]), int(r[1]), r[2], r[3]) for r in data]


# Last mapping returned by `get_node_to_path_and_type` (read many times per evaluation)
_NODE_TO_PATH_CACHE = {}


def get_node_to_path_and_type(cfg):
    """Returns {index_id: {"path", "type", ...}}. The result is shared: do not modify it.

    `path` is the node shown in evaluation outputs (see `NodeTypeSchema.display`), and the
    other keys are the raw attributes listed in `NodeTypeSchema.display_extra` (e.g. `cmd`).

    Only the nodes of the built graphs are loaded (6.35M of the 44.5M nodes of TRACE_E3);
    other nodes are read from the database when accessed (see `NodeToPathType`).
    """
    out_path = cfg.construction._node_id_to_path
    out_file = os.path.join(out_path, "node_to_paths.pkl")
    if out_file in _NODE_TO_PATH_CACHE:
        return _NODE_TO_PATH_CACHE[out_file]
    _NODE_TO_PATH_CACHE.clear()  # only keep the last dataset in memory
    node_to_path_type = NodeToPathType(cfg, _compute_node_to_path_and_type(cfg, out_path, out_file))
    _NODE_TO_PATH_CACHE[out_file] = node_to_path_type
    return node_to_path_type


def _node_info(nt, row):
    """{"path", "type", ...} of a node of type `nt` from its row (index_id, *nt.columns)."""
    raw_attrs = {attr: value for (attr, _), value in zip(nt.columns, row[1:])}
    info = {"path": nt.display_path(raw_attrs), "type": nt.name}
    for key, attr in nt.display_extra:
        info[key] = raw_attrs[attr]
    return info


class NodeToPathType(dict):
    """{index_id: info} of `get_node_to_path_and_type`, where the nodes that are not in it
    (nodes absent from the graphs) are read from the database on access, by `[]` or `get`.
    Iterating it only covers the loaded nodes."""

    def __init__(self, cfg, data):
        super().__init__(data)
        self._cfg = cfg
        self._absent = set()

    def __missing__(self, index_id):
        info = self._read(index_id)
        if info is None:
            raise KeyError(index_id)
        return info

    def get(self, index_id, default=None):
        if index_id in self:
            return self[index_id]
        info = self._read(index_id)
        return default if info is None else info

    def _read(self, index_id):
        if index_id in self._absent or isinstance(index_id, bool):
            return None
        try:
            index_id = int(index_id)
        except (TypeError, ValueError):
            return None
        if index_id in self:
            return dict.__getitem__(self, index_id)
        cur, connect = init_database_connection(self._cfg)
        try:
            for nt in get_schema(self._cfg).node_types:
                columns = ", ".join(["index_id"] + [col for _, col in nt.columns])
                cur.execute(f"SELECT {columns} FROM {nt.table} WHERE index_id = %s;", (index_id,))
                row = cur.fetchone()
                if row is not None:
                    info = _node_info(nt, row)
                    dict.__setitem__(self, index_id, info)
                    return info
        finally:
            connect.close()
        self._absent.add(index_id)
        return None

    def __reduce__(self):  # pickled as a plain dict
        return (dict, (dict(self),))


def _graph_node_ids(cfg):
    """Index IDs of the nodes of the built graphs (from split2nodes), or None if unknown."""
    path = os.path.join(cfg.construction._dicts_dir, "split2nodes.pkl")
    if not os.path.exists(path):
        return None
    split2nodes = torch.load(path)
    return {int(n) for nodes in split2nodes.values() for n in nodes}


def _compute_node_to_path_and_type(cfg, out_path, out_file):
    if os.path.exists(out_file):
        return torch.load(out_file)

    os.makedirs(out_path, exist_ok=True)
    graph_nodes = _graph_node_ids(cfg)
    cur, connect = init_database_connection(cfg)
    node_to_path_type = {}
    try:
        for nt in get_schema(cfg).node_types:
            columns = ", ".join(["index_id"] + [col for _, col in nt.columns])
            for row in _iter_rows(connect, f"SELECT {columns} FROM {nt.table};"):
                if graph_nodes is None or int(row[0]) in graph_nodes:
                    node_to_path_type[row[0]] = _node_info(nt, row)
    finally:
        connect.close()

    torch.save(node_to_path_type, out_file)
    return node_to_path_type


def selected_edge_attribute_names(value, dataset_name):
    """Names selected by `feat_inference.edge_attributes` for a dataset (None: all).

    `value` is `all`, a comma-separated list of names, or per-dataset lists
    `DATASET:a,b;OTHER_DATASET:c` (datasets that are not listed use all their attributes).
    """
    value = (value or "all").strip()
    if ":" in value:
        per_dataset = {}
        for entry in value.split(";"):
            if entry.strip():
                dataset, _, names = entry.partition(":")
                per_dataset[dataset.strip().upper()] = names
        value = per_dataset.get(dataset_name.upper(), "all").strip()
    if value == "all":
        return None
    return {n.strip() for n in value.split(",") if n.strip()}


def get_input_edge_attributes(cfg, selected=True):
    """Edge attributes of the dataset used as model inputs: all but the one used as edge type,
    restricted to `feat_inference.edge_attributes` when `selected`."""
    schema = get_schema(cfg)
    attributes = [a for a in schema.edge_attributes if a.name != schema.edge_type_attribute]
    if selected:
        chosen = getattr(getattr(cfg, "feat_inference", None), "edge_attributes", None)
        names = selected_edge_attribute_names(chosen, cfg.dataset.name)
        if names is not None:
            attributes = [a for a in attributes if a.name in names]
    return attributes


def uses_aggregated_edge_attributes(cfg):
    """Whether merged edges carry aggregated attributes (`merge_edges.aggregate_attributes`)."""
    transformation = getattr(cfg, "transformation", None)
    if transformation is None or "merge_edges" not in (transformation.used_methods or ""):
        return False
    return bool(getattr(transformation.merge_edges, "aggregate_attributes", False))


def aggregated_edge_attribute_columns(cfg, selected=True):
    """Names of the aggregated attributes of merged edges (see `merge_edges.aggregate_attributes`):
    {attribute name: [(column, value or statistic)]}, plus the number of merged events (`count`)."""
    columns = {"count": [("count", None)]}
    for attr in get_input_edge_attributes(cfg, selected=selected):
        if attr.kind == "categorical":
            columns[attr.name] = [(f"{attr.name}={value}", value) for value in attr.values]
        else:
            columns[attr.name] = [(f"{attr.name}:{stat}", stat) for stat in ("n", "mean", "std")]
    return columns


def get_edge_attrs_dim(cfg):
    """Size of the edge attribute features built by `encode_edge_attributes`."""
    if uses_aggregated_edge_attributes(cfg):
        return sum(len(cols) for cols in aggregated_edge_attribute_columns(cfg).values())
    return sum(
        len(a.values) if a.kind == "categorical" else 1 for a in get_input_edge_attributes(cfg)
    )


def standardize_counts(x):
    """Standardization of the edge features of Euler, Argus and LARES, per column of an (E, d)
    tensor: sigmoid(floor(floor(x) / std))."""
    x = torch.floor(x)
    std = x.std(dim=0, unbiased=False) if x.shape[0] > 1 else torch.zeros(x.shape[1])
    return torch.sigmoid(torch.floor(x / (std + 1e-6)))


def encode_aggregated_edge_attributes(cfg, columns):
    """Features of the aggregated attributes of merged edges, as an (E, get_edge_attrs_dim) tensor.

    Args:
        columns: {column of `aggregated_edge_attribute_columns`: list of the values of the edges}.
    """
    names = [c for cols in aggregated_edge_attribute_columns(cfg).values() for c, _ in cols]
    x = torch.tensor(
        [[0.0 if v is None else float(v) for v in columns[name]] for name in names],
        dtype=torch.float,
    ).T
    x = torch.nan_to_num(x, nan=0.0).clamp(min=0)
    return standardize_counts(x)


def encode_edge_attributes(cfg, columns):
    """Features of the edge attributes of a graph, as a (E, get_edge_attrs_dim) float tensor.

    Categorical attributes are one-hot encoded (values outside the vocabulary are `OTHER`),
    numeric attributes are log1p-scaled (negative and missing values are 0).

    Args:
        columns: {attribute name: list of the values of the edges}, as stored in the graphs.

    Returns:
        The tensor, or None if the dataset has no input edge attribute.
    """
    attributes = get_input_edge_attributes(cfg)
    if not attributes:
        return None
    feats = []
    for attr in attributes:
        values = columns[attr.name]
        if attr.kind == "categorical":
            value2idx = {v: i for i, v in enumerate(attr.values)}
            other = value2idx[attr.OTHER]
            idx = torch.tensor([value2idx.get(str(v), other) for v in values], dtype=torch.long)
            feats.append(torch.nn.functional.one_hot(idx, num_classes=len(attr.values)).float())
        else:
            x = torch.tensor(
                [float("nan") if v is None else float(v) for v in values], dtype=torch.float
            )
            x = torch.nan_to_num(x, nan=0.0).clamp(min=0)
            feats.append(torch.log1p(x).unsqueeze(1))
    return torch.cat(feats, dim=1)
