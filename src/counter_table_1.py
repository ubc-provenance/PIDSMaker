from provnet_utils import *
from config import *

import numpy as np
from labelling import get_ground_truth, get_events
from detection.evaluation_methods.evaluation_utils import compute_tw_labels, get_start_end_from_graph
import networkx as nx
import datetime

def select_events(cur,
                  start_time: str,
                  end_time: str,
                  ):

    sql = f"SELECT * FROM event_table WHERE timestamp_rec > {start_time} AND timestamp_rec < {end_time} ORDER BY timestamp_rec ;"

    cur.execute(sql)
    records = cur.fetchall()

    event2msg = {}

    for i in tqdm(records,desc='Selecting events'):
        src_index_id = str(i[1])
        operation = i[2]
        dst_index_id = str(i[4])
        event_uuid = i[5]
        timestamp = i[6]
        event_id = i[7]

        event2msg[event_uuid] = [src_index_id, operation, dst_index_id, timestamp, event_id]

    return event2msg

def gen_graph(start_time, end_time, cfg):
    cur, connect = init_database_connection(cfg)
    event2msg = select_events(cur, start_time, end_time)

    print(f"generate graph")
    edge_list = []
    for k, v in event2msg.items():
        src = v[0]
        dst = v[2]
        if src != dst:
            edge_list.append((src, dst))

    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)

    return graph

def get_n_hop_neighbors(graph, node, n):

    neighbors = set()
    current_level = {node}

    for _ in range(n):
        next_level = set()
        for current_node in current_level:
            next_level.update(graph.successors(current_node))
            next_level.update(graph.predecessors(current_node))
        neighbors.update(next_level)
        current_level = next_level

    neighbors.add(node)

    return neighbors

def compute_node_number_tw(split_files):
    num_nodes = []
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    for graph_path in tqdm(sorted_paths, desc='Computing node number'):
        graph = torch.load(graph_path)
        num_nodes.append(len(graph.nodes()))
    return num_nodes

def get_n_hop_of_GP(graph, GPs, n):
    n_hop_of_GP = set()
    for gp in GPs:
        n_hop_of_GP.add(gp)

    for nid in graph.nodes():
        if nid in GPs:
            neighbors = get_n_hop_neighbors(graph, nid, n)
            n_hop_of_GP |= neighbors

    return n_hop_of_GP

def ns_time_to_datetime_US(nanoseconds):
    """
    :param nanoseconds: int   纳秒级时间戳
    :return: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    """
    # 将纳秒级时间戳转换为秒
    seconds = nanoseconds / 1e9
    # 将秒级时间戳转换为 UTC 时间的 datetime 对象
    dt_utc = datetime.datetime.utcfromtimestamp(seconds)
    # 将 UTC 时间转换为美国东部时间（US/Eastern）
    tz = pytz.timezone('US/Eastern')
    dt_us = dt_utc.replace(tzinfo=pytz.utc).astimezone(tz)
    # 格式化为字符串
    date_str = dt_us.strftime("%Y-%m-%d %H:%M:%S")
    return date_str


def main(cfg):
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    cur, connect = init_database_connection(cfg)

    # # counter max, min, median, mean of node number
    # node_num_train = compute_node_number_tw(split_files=cfg.dataset.train_files)
    # node_num_val = compute_node_number_tw(split_files=cfg.dataset.val_files)
    # node_num_test = compute_node_number_tw(split_files=cfg.dataset.test_files)
    # node_num_unused = compute_node_number_tw(split_files=cfg.dataset.unused_files)
    #
    # num_in_all_tw = node_num_train + node_num_val + node_num_test + node_num_unused
    # min_value = min(num_in_all_tw)
    # max_value = max(num_in_all_tw)
    # median_value = np.median(num_in_all_tw)
    # average_value = np.mean(num_in_all_tw)
    # print(f"Number of nodes in all tw:")
    # print(f"Min: {min_value}")
    # print(f"Median: {median_value}")
    # print(f"Max: {max_value}")
    # print(f"Mean: {average_value}")

    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [str(nid) for nid in GP_nids]
    print(f"There are {len(GPs)} malicious nodes")

    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)

    # test_graph = torch.load(sorted_paths[0])
    # for u, v, k, data in test_graph.edges(keys=True,data=True):
    #     print(f"u: {u}, v: {v}, k: {k}, data: {data}")

    ## counter n-hop neighbors of malicious nodes
    # nhop = 2
    # all_n_hops = set()
    # for graph_path in tqdm(sorted_paths, desc='Computing 2-hop neighbors of GPs'):
    #     graph = torch.load(graph_path)
    #     n_hop_of_GP = get_n_hop_of_GP(graph, GPs, nhop)
    #     all_n_hops |= n_hop_of_GP
    # num_in_nhop = len(all_n_hops)
    # print(f"Number of nodes in {nhop} hop neighbors of GPs: {num_in_nhop}")

    ## counter n-hop neighbors of malicious nodes only in attack-relevant tws
    # nhop = 2
    # all_n_hops = set()
    # total_number_in_nhop = 0
    # for tw, nid2count in tw_to_malicious_nodes.items():
    #         if len(nid2count.items()) > 0:
    #             graph = torch.load(sorted_paths[tw])
    #             n_hop_of_GP = get_n_hop_of_GP(graph, GPs, nhop)
    #             all_n_hops |= n_hop_of_GP
    #             total_number_in_nhop += len(n_hop_of_GP)
    # num_in_nhop = len(all_n_hops)
    # print(f"Number of unique nodes in {nhop} hop neighbors of GPs: {num_in_nhop}")
    # print(f"Number of nodes in {nhop} hop neighbors of GPs: {total_number_in_nhop}")


    # count total number of nodes in attack relevant graphs
    print(f"Number of GPs: {len(GPs)}")
    unique_nodes_in_attack_tw = set()
    node_number_in_attak_tw = 0
    for tw, nid2count in tw_to_malicious_nodes.items():
        if len(nid2count.items()) > 0:
            graph = torch.load(sorted_paths[tw])
            unique_nodes_in_attack_tw |= set(graph.nodes())
            node_number_in_attak_tw += len(graph.nodes())
    print(f"total number of nodes in attack relevant graphs: {node_number_in_attak_tw}")
    print(f"total number of unique nodes in attack relevant graphs: {len(unique_nodes_in_attack_tw)}")

    ## check malicious tws
    # for tw, nid2count in tw_to_malicious_nodes.items():
    #     if len(nid2count.items()) > 0:
    #         graph = torch.load(sorted_paths[tw])
    #         start, end = get_start_end_from_graph(graph)
    #         print(f"For malicious time window {tw}:")
    #         print(f"Time range is  {ns_time_to_datetime_US(start)} -> {ns_time_to_datetime_US(end)}")
    #         print(f"graph file name is {sorted_paths[tw]}")
    #
    #         n_set = set()
    #         rows = get_events(cur, start, end)
    #         for row in rows:
    #             src_id = row[1]
    #             dst_id = row[4]
    #             n_set.add(str(src_id))
    #             n_set.add(str(dst_id))
    #
    #         print(f"Number of nodes in the graph: {len(graph.nodes())}")
    #         print(f"Number of nodes in the time range: {len(n_set)}")



    # nhop = 2
    # all_n_hops = set()
    # total_number_in_nhop = 0
    # start_time = '1523478987898404334'
    # end_time = '1523586247675522567'
    # start_time_str = ns_to_us_time_string_24h(int(start_time))
    # end_time_str = ns_to_us_time_string_24h(int(end_time))
    # print(f"start time is {start_time_str}; end time is {end_time_str}")
    #
    # graph = gen_graph(start_time, end_time, cfg)
    # n_hop_of_GP = get_n_hop_of_GP(graph, GPs, nhop)
    # all_n_hops |= n_hop_of_GP
    # total_number_in_nhop += len(n_hop_of_GP)
    # num_in_nhop = len(all_n_hops)
    # print(f"Number of unique nodes in {nhop} hop neighbors of GPs: {num_in_nhop}")
    # print(f"Number of nodes in {nhop} hop neighbors of GPs: {total_number_in_nhop}")





if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)