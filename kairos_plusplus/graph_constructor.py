import logging
import os
from datetime import datetime, timedelta
import networkx as nx

from config import *
from provnet_utils import *


def get_node_list(cur):
    # node hash id to node label and type
    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}

    # hash_id | node_type | msg | index_id
    for i in rows:
        nodeid2msg[i[0]] = [i[1], i[2]]

    return nodeid2msg

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

def gen_edge_fused_tw(cur, nodeid2msg, logger, cfg):

    include_edge_type = rel2id

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i:i + batch_size]

    start, end = cfg.dataset.start_end_day_range
    for day in range(start, end):
        date_start = cfg.dataset.year_month + '-' + str(day) + ' 00:00:00'
        date_stop = cfg.dataset.year_month + '-' + str(day + 1) + ' 00:00:00'

        timestamps = [date_start, date_stop]

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
            for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(events):
                if operation in include_edge_type:
                    event_tuple = (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
                    events_list.append(event_tuple)

            start_time = events_list[0][-2]
            temp_list = []
            for batch_edges in get_batches(events_list, BATCH):
                for j in batch_edges:
                    temp_list.append(j)

                if batch_edges[-1][-2] > start_time + time_window_size:
                    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(
                        batch_edges[-1][-2])

                    logger.info(f"Start create edge fused time window graph for {time_interval}")

                    node_info = {}
                    edge_info = {}
                    for (
                    src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
                            temp_list, desc=f"edge fused graph for time window {time_interval}"):
                        if src_index_id not in node_info:
                            node_type, label = nodeid2msg[src_node]
                            node_info[src_index_id] = {
                                'label': label,
                                'node_type': node_type,
                            }
                        if dst_index_id not in node_info:
                            node_type, label = nodeid2msg[dst_node]
                            node_info[dst_index_id] = {
                                'label': label,
                                'node_type': node_type,
                            }

                        if (src_index_id, dst_index_id) not in edge_info:
                            edge_info[(src_index_id, dst_index_id)] = []

                        edge_info[(src_index_id, dst_index_id)].append((timestamp_rec, operation, event_uuid))

                    edge_list = []

                    for (src, dst), data in edge_info.items():
                        sorted_data = sorted(data, key=lambda x:x[0])
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

                    logger.info(f"Start creating graph for {time_interval}")
                    graph = nx.MultiDiGraph()

                    for node, info in node_info.items():
                        graph.add_node(
                            node,
                            node_type=info['node_type'],
                            label=info['label']
                        )

                    for edge in edge_list:
                        graph.add_edge(
                            edge['src'],
                            edge['dst'],
                            event_uuid=edge['event_uuid'],
                            time=edge['time'],
                            label=edge['label']
                        )

                    date_dir = f"{cfg.preprocessing._graphs_dir}/graph_5_{day}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    logger.info(f"Saving graph for {time_interval}")
                    torch.save(graph, graph_name)

                    logger.info(f"[{time_interval}] Num of edges: {len(edge_list)}")
                    logger.info(f"[{time_interval}] Num of events: {len(temp_list)}")
                    logger.info(f"[{time_interval}] Num of nodes: {len(node_info.keys())}")
                    start_time = batch_edges[-1][-2]
                    temp_list.clear()


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    logger = get_logger(
        name="graph_construction_edge_fused_tw",
        filename=os.path.join(cfg.preprocessing._logs_dir, "edge_fused_tw_graph.log"))

    cur, connect = init_database_connection()
    nodeid2msg = get_node_list(cur=cur)

    gen_edge_fused_tw(cur=cur, nodeid2msg=nodeid2msg, logger=logger, cfg=cfg)
