from collections import defaultdict
import logging
import os
from datetime import datetime, timedelta
import networkx as nx
import torch
from config import *
from provnet_utils import *


def compute_indexid2msg(cfg):
    """
    Returns a dict that maps {
        node => [node type, feature msg],
    }
    """
    cur, connect = init_database_connection(cfg)
    
    use_hashed_label = cfg.preprocessing.build_graphs.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    indexid2msg = {}
    
    def get_label_str_from_features(attrs, node_type):
        label_str = ' '.join([attrs[label_used] for label_used in node_label_features[node_type]])
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of netflow nodes: {len(records)}")

    for i in records:
        attrs = {
            'type': 'netflow',
            'local_ip': str(i[2]),
            'local_port': str(i[3]),
            'remote_ip': str(i[4]),
            'remote_port': str(i[5])
        }
        index_id = str(i[-1])
        node_type = attrs['type']
        label_str = get_label_str_from_features(attrs, node_type)
            
        indexid2msg[index_id] = [node_type, label_str]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of process nodes: {len(records)}")

    for i in records:
        attrs = {
            'type': 'subject',
            'path': str(i[2]),
            'cmd_line': str(i[3])
        }
        index_id = str(i[-1])
        node_type = attrs['type']
        label_str = get_label_str_from_features(attrs, node_type)
            
        indexid2msg[index_id] = [node_type, label_str]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of file nodes: {len(records)}")

    for i in records:
        attrs = {
            'type': 'file',
            'path': str(i[2])
        }
        index_id = str(i[-1])
        node_type = attrs['type']
        label_str = get_label_str_from_features(attrs, node_type)
            
        indexid2msg[index_id] = [node_type, label_str]

    return indexid2msg #{index_id: [node_type, msg]}

def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """
    The saving must occur after the graph construction, because some edge types
    are not considered and this results in some nodes that are not used in the pipeline.
    These nodes must be removed before storing to disk to avoid future errors.
    """
    all_nodes = set().union(*(split2nodes[split] for split in ["train", "val", "test"]))
    indexid2msg = {k: v for k, v in indexid2msg.items() if k in all_nodes}
    
    out_dir = cfg.preprocessing.build_graphs._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))

def compute_and_save_split2nodes(cfg):
    """
    Returns a dict that maps {
        "train" => nodes in train,
        "test" => nodes in test,
        "val" => nodes in val,
    }
    """
    split_to_files = get_split_to_files(cfg, cfg.preprocessing.build_graphs._graphs_dir)
    split2nodes = defaultdict(set)
    
    for split, files in split_to_files.items():
        graph_list = [torch.load(path) for path in files]
        for G in log_tqdm(graph_list, desc=f"Check nodes in {split} set"):
            for node in G.nodes():
                split2nodes[split].add(node)
    split2nodes = dict(split2nodes)
    
    out_dir = cfg.preprocessing.build_graphs._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))
    
    return split2nodes

def generate_timestamps(start_time, end_time, interval_minutes):
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    timestamps = []
    current_time = start
    while current_time <= end:
        timestamps.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=interval_minutes)
    timestamps.append(end)
    return timestamps


def gen_edge_fused_tw(indexid2msg, cfg):
    cur, connect = init_database_connection(cfg)
    rel2id = get_rel2id(cfg)
    include_edge_type = rel2id

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i:i + batch_size]

    # In test mode, we ensure to get 1 TW in each set
    days = get_days_from_cfg(cfg)

    for day in days:
        date_start = cfg.dataset.year_month + '-' + str(day) + ' 00:00:00'
        date_stop = cfg.dataset.year_month + '-' + str(day + 1) + ' 00:00:00'

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start)
            end_ns_timestamp = datetime_to_ns_time_US(stop)
            sql = """
            select * from event_table
            where
                  timestamp_rec>'%s' and timestamp_rec<'%s'
                   ORDER BY timestamp_rec;
            """ % (start_ns_timestamp, end_ns_timestamp)
            cur.execute(sql)
            events = cur.fetchall()

            if len(events) == 0:
                continue

            events_list = []
            for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in events:
                if operation in include_edge_type:
                    event_tuple = (
                    src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
                    events_list.append(event_tuple)

            start_time = events_list[0][-2]
            temp_list = []
            BATCH = 1024
            window_size_in_sec = cfg.preprocessing.build_graphs.time_window_size * 60_000_000_000

            last_batch = False
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
                    last_batch = True

                if (batch_edges[-1][-2] > start_time + window_size_in_sec) or last_batch:
                    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(
                        batch_edges[-1][-2])

                    log(f"Start create edge fused time window graph for {time_interval}")

                    node_info = {}
                    edge_info = {}
                    for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in temp_list:
                        if src_index_id not in node_info:
                            node_type, label = indexid2msg[src_index_id]
                            node_info[src_index_id] = {
                                'label': label,
                                'node_type': node_type,
                            }
                        if dst_index_id not in node_info:
                            node_type, label = indexid2msg[dst_index_id]
                            node_info[dst_index_id] = {
                                'label': label,
                                'node_type': node_type,
                            }

                        if (src_index_id, dst_index_id) not in edge_info:
                            edge_info[(src_index_id, dst_index_id)] = []

                        edge_info[(src_index_id, dst_index_id)].append((timestamp_rec, operation, event_uuid))

                    edge_list = []

                    for (src, dst), data in edge_info.items():
                        sorted_data = sorted(data, key=lambda x: x[0])
                        operation_list = [entry[1] for entry in sorted_data]

                        indices = []
                        current_type = None
                        current_start_index = None

                        for idx, item in enumerate(operation_list):
                            if item == current_type:
                                continue
                            else:
                                if current_type is not None and current_start_index is not None:
                                    indices.append(current_start_index)
                                current_type = item
                                current_start_index = idx

                        if current_type is not None and current_start_index is not None:
                            indices.append(current_start_index)

                        for k in indices:
                            edge_list.append({
                                'src': src,
                                'dst': dst,
                                'time': sorted_data[k][0],
                                'label': sorted_data[k][1],
                                'event_uuid': sorted_data[k][2]
                            })

                    log(f"Start creating graph for {time_interval}")
                    graph = nx.MultiDiGraph()

                    for node, info in node_info.items():
                        graph.add_node(
                            node,
                            node_type=info['node_type'],
                            label=info['label']
                        )

                    for i, edge in enumerate(edge_list):
                        graph.add_edge(
                            edge['src'],
                            edge['dst'],
                            event_uuid=edge['event_uuid'],
                            time=edge['time'],
                            label=edge['label']
                        )

                        # For unit tests, we only want few edges
                        NUM_TEST_EDGES = 2000
                        if cfg._test_mode and i >= NUM_TEST_EDGES:
                            break

                    date_dir = f"{cfg.preprocessing.build_graphs._graphs_dir}/graph_{day}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    log(f"Saving graph for {time_interval}")
                    torch.save(graph, graph_name)

                    log(f"[{time_interval}] Num of edges: {len(edge_list)}")
                    log(f"[{time_interval}] Num of events: {len(temp_list)}")
                    log(f"[{time_interval}] Num of nodes: {len(node_info.keys())}")
                    start_time = batch_edges[-1][-2]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        break


def main(cfg):
    log_start(__file__)
    
    indexid2msg = compute_indexid2msg(cfg=cfg)

    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg)
    
    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
