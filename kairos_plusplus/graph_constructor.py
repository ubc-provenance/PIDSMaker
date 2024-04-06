import logging
import os
from datetime import datetime, timedelta
import networkx as nx
import argparse

from config import *
from provnet_utils import *

logger = logging.getLogger("parser")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'parser.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

def gen_kairos_date_graph(cur, nodeid2msg, graphs_dir, time_window_size=None):
    include_edge_type = rel2id

    os.system("mkdir -p {}".format(graphs_dir))
    for day in range(8, 18):
        date_start = '2019-05-' + str(day) + ' 00:00:00'
        date_stop = '2019-05-' + str(day + 1) + ' 00:00:00'

        if time_window_size != None:
            timestamps = generate_timestamps(start_time=date_start,
                                             end_time=date_stop,
                                             interval_minutes=time_window_size)
        else:
            timestamps = [date_start, date_stop]

        for i in range(0, len(timestamps)-1):
            start = timestamps[i]
            stop = timestamps[i+1]
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

            # node_list = set()
            # for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
            #         events):
            #     if operation in include_edge_type:
            #         node_list.add(src_node)
            #         node_list.add(dst_node)
            #
            # events_list = []
            # for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
            #         events):
            #     if src_node in node_list and dst_node in node_list:
            #         # The Kairos codes don't consider interactions between processes, so we add those interactions into Kairos graphs as well
            #         if operation in include_edge_type or ( nodeid2msg[src_node][0] == 'subject' and nodeid2msg[dst_node][0] == 'subject'):
            #             i = (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
            #             events_list.append(i)

            events_list = []
            for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
                    events):
                if operation in include_edge_type:
                    i = (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
                    events_list.append(i)

            graph = nx.MultiDiGraph()
            for src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id in tqdm(events_list, desc='2018-04-' + str(day)):
                if src_index_id not in graph.nodes:
                    node_type, label = nodeid2msg[src_node]
                    graph.add_node(
                        src_index_id,
                        node_type=node_type,
                        label=label
                    )
                if dst_index_id not in graph.nodes:
                    node_type, label = nodeid2msg[dst_node]
                    graph.add_node(
                        dst_index_id,
                        node_type=node_type,
                        label=label
                    )
                graph.add_edge(
                    src_index_id,
                    dst_index_id,
                    event_uuid=event_uuid,
                    time=timestamp_rec,
                    label=operation
                )
            # e.g. e3-graph-02-00_00_00
            graph_name = f"e5-graph-{str(day).zfill(2)}-{start[-8:].replace(':','_')}"
            torch.save(graph, graphs_dir + graph_name)

def gen_kairos_time_window_graph(cur, nodeid2msg, graphs_dir):
    include_edge_type = rel2id

    def get_batches(arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i:i + batch_size]

    for day in testing_date:
        date_start = '2019-05-' + str(day) + ' 00:00:00'
        date_stop = '2019-05-' + str(day + 1) + ' 00:00:00'

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

            # node_list = set()
            # for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
            #         events):
            #     if operation in include_edge_type:
            #         node_list.add(src_node)
            #         node_list.add(dst_node)
            #
            # events_list = []
            # for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(
            #         events):
            #     if src_node in node_list and dst_node in node_list:
            #         # The Kairos codes don't consider interactions between processes, so we add those interactions into Kairos graphs as well
            #         if operation in include_edge_type or (nodeid2msg[src_node][0] == 'subject' and nodeid2msg[dst_node][0] == 'subject'):
            #             i = (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
            #             events_list.append(i)

            events_list = []
            for (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id) in tqdm(events):
                if operation in include_edge_type:
                    i = (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id)
                    events_list.append(i)

            logger.info(f"Before edge filtering: {len(events)}. After edge filtering: {len(events_list)}")

            start_time = events_list[0][-2]
            temp_list = []
            for batch_edges in get_batches(events_list, BATCH):
                for i in batch_edges:
                    temp_list.append(i)

                if batch_edges[-1][-2] > start_time + time_window_size:
                    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(
                        batch_edges[-1][-2])

                    graph = nx.MultiDiGraph()
                    for src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec, _id in tqdm(
                            temp_list, desc=time_interval):

                        if src_index_id not in graph.nodes:
                            node_type, label = nodeid2msg[src_node]
                            graph.add_node(
                                src_index_id,
                                node_type=node_type,
                                label=label
                            )
                        if dst_index_id not in graph.nodes:
                            node_type, label = nodeid2msg[dst_node]
                            graph.add_node(
                                dst_index_id,
                                node_type=node_type,
                                label=label
                            )
                        graph.add_edge(
                            src_index_id,
                            dst_index_id,
                            event_uuid=event_uuid,
                            time=timestamp_rec,
                            label=operation
                        )


                    date_dir = f"{graphs_dir}/graph_5_{day}/"
                    os.system(f"mkdir -p {date_dir}")
                    graph_name = f"{date_dir}/{time_interval}"
                    torch.save(graph, graph_name)

                    logger.info(f"[{time_interval}] Num of event: {len(temp_list)}")
                    start_time = batch_edges[-1][-2]
                    temp_list.clear()


if __name__ == "__main__":
    os.system(f"mkdir -p {graphs_dir}")

    cur, connect = init_database_connection()
    nodeid2msg = get_node_list(cur=cur)

    global testing_date
    testing_date = [14, 15]

    gen_kairos_date_graph(cur=cur, nodeid2msg=nodeid2msg, graphs_dir=graphs_dir)

    gen_kairos_time_window_graph(cur=cur, nodeid2msg=nodeid2msg, graphs_dir=graphs_dir)




