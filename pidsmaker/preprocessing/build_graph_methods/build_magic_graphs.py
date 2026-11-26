import json
import os
from datetime import datetime, timedelta

import networkx as nx

from pidsmaker.config import get_dates_from_cfg, get_node_feats_from_cfg
from pidsmaker.utils.database import events_order_sql
from pidsmaker.utils.dataset_utils import get_node_map, get_rel2id, get_schema, iter_nodes
from pidsmaker.utils.graph_store import save_graph
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    init_database_connection,
    log,
    log_start,
    ns_time_to_datetime_US,
)


_NONE_VALUES = {"None", "NA", "0", "none", "null", ""}


def _build_label(attrs, node_type, node_label_features, use_hashed_label):
    feats = node_label_features[node_type]
    if feats == ["auto"]:
        parts = [attrs["type"]]
        for key in sorted(attrs):
            if key == "type":
                continue
            val = attrs[key]
            if val not in _NONE_VALUES:
                parts.append(val)
        label_str = " ".join(parts)
    else:
        label_str = " ".join([attrs[f] for f in feats])
    if use_hashed_label:
        from pidsmaker.utils.utils import stringtomd5
        label_str = stringtomd5(label_str)
    return label_str


def get_node_list(cur, cfg):
    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_node_feats_from_cfg(cfg)

    uuid2idx = {}
    uuid2type = {}
    uuid2name = {}
    hash2uuid = {}

    for node_type, node_uuid, hash_id, index_id, attrs in iter_nodes(cfg, cur.connection):
        label_str = _build_label(attrs, node_type, node_label_features, use_hashed_label)

        uuid2idx[node_uuid] = int(index_id)
        uuid2type[node_uuid] = node_type
        uuid2name[node_uuid] = label_str
        hash2uuid[hash_id] = node_uuid

    return uuid2idx, uuid2type, uuid2name, hash2uuid


def generate_graphs(cur, uuid2type, graph_out_dir, hash2uuid, cfg):
    rel2id = get_rel2id(cfg)
    ntype2id = get_node_map(cfg)
    include_edge_type = rel2id
    node_type_dict = ntype2id
    operations_sql = get_schema(cfg).operations_sql()

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]

    # In test mode, we ensure to get 1 TW in each set
    dates = get_dates_from_cfg(cfg)

    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start, timezone=cfg.dataset.timezone)
            end_ns_timestamp = datetime_to_ns_time_US(stop, timezone=cfg.dataset.timezone)
            sql = """
            select * from event_table
            where
                  timestamp_rec>'%s' and timestamp_rec<'%s'%s
                   %s;
            """ % (
                start_ns_timestamp,
                end_ns_timestamp,
                operations_sql,
                events_order_sql(cur, "timestamp_rec"),
            )
            cur.execute(sql)
            events = cur.fetchall()

            if len(events) == 0:
                continue

            events_list = []
            for (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in events:
                if operation in include_edge_type:
                    event_tuple = (
                        src_node,
                        src_index_id,
                        operation,
                        dst_node,
                        dst_index_id,
                        event_uuid,
                        timestamp_rec,
                        _id,
                    )
                    events_list.append(event_tuple)

            start_time = events_list[0][-2]
            temp_list = []
            BATCH = 1024
            window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

            last_batch = False
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
                    last_batch = True

                if (batch_edges[-1][-2] > start_time + window_size_in_ns) or last_batch:
                    time_interval = (
                        ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone)
                        + "~"
                        + ns_time_to_datetime_US(batch_edges[-1][-2], timezone=cfg.dataset.timezone)
                    )

                    log(f"Start create edge fused time window graph for {time_interval}")

                    g = nx.DiGraph()
                    node_visited = set()

                    for event_tuple in events_list:
                        (
                            src_node,
                            src_index_id,
                            operation,
                            dst_node,
                            dst_index_id,
                            event_uuid,
                            timestamp_rec,
                            _id,
                        ) = event_tuple
                        if src_index_id not in node_visited:
                            g.add_node(
                                int(src_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[src_node]]],
                            )
                            node_visited.add(src_index_id)
                        if dst_index_id not in node_visited:
                            g.add_node(
                                int(dst_index_id),
                                type=node_type_dict[uuid2type[hash2uuid[dst_node]]],
                            )
                            node_visited.add(dst_index_id)
                        if not g.has_edge(int(src_index_id), int(dst_index_id)):
                            g.add_edge(
                                int(src_index_id),
                                int(dst_index_id),
                                type=include_edge_type[operation],
                            )

                    date_dir = f"{graph_out_dir}/graph_{date}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    print(f"Saving graph for {time_interval}")
                    save_graph(g, graph_name)

                    start_time = batch_edges[-1][-2]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break
    return


def main(cfg):
    log_start(__file__)
    cur, connect = init_database_connection(cfg)
    uuid2idx, uuid2type, uuid2name, hash2uuid = get_node_list(cur=cur, cfg=cfg)

    os.makedirs(cfg.construction._magic_dir, exist_ok=True)
    os.makedirs(cfg.construction._magic_graphs_dir, exist_ok=True)
    file_out_dir = cfg.construction._magic_dir
    graph_out_dir = cfg.construction._magic_graphs_dir

    with open(os.path.join(file_out_dir, "names.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2name, fw)
    with open(os.path.join(file_out_dir, "types.json"), "w", encoding="utf-8") as fw:
        json.dump(uuid2type, fw)

    generate_graphs(
        cur=cur, uuid2type=uuid2type, graph_out_dir=graph_out_dir, hash2uuid=hash2uuid, cfg=cfg
    )

    del uuid2idx, uuid2type, uuid2name, hash2uuid
