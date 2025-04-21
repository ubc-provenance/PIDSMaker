from datetime import datetime

import networkx as nx
import pytz
import tqdm

from pidsmaker.labelling import get_ground_truth
from pidsmaker.provnet_utils import (
    datetime_to_ns_time_US,
    get_runtime_required_args,
    get_yml_cfg,
    init_database_connection,
    log,
)

dataset_to_time = {
    "CLEARSCOPE_E5": [
        ("2019-05-15 00:00:00", "2019-05-16 00:00:00"),
        ("2019-05-17 00:00:00", "2019-05-18 00:00:00"),
    ],
    "CLEARSCOPE_E3": [("2018-04-11 00:00:00", "2018-04-13 00:00:00")],
}


def select_events(
    cur,
    start_time: str,
    end_time: str,
):
    sql = f"SELECT * FROM event_table WHERE timestamp_rec > {start_time} AND timestamp_rec < {end_time} ORDER BY timestamp_rec ;"

    cur.execute(sql)
    records = cur.fetchall()

    event2msg = {}

    for i in tqdm(records, desc="Selecting events"):
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

    print("generate graph")
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
    tz = pytz.timezone("US/Eastern")
    dt_us = dt_utc.replace(tzinfo=pytz.utc).astimezone(tz)
    # 格式化为字符串
    date_str = dt_us.strftime("%Y-%m-%d %H:%M:%S")
    return date_str


def get_n_hop_of_GP(graph, GPs, n):
    n_hop_of_GP = set()
    for gp in GPs:
        n_hop_of_GP.add(gp)

    for nid in graph.nodes():
        if nid in GPs:
            neighbors = get_n_hop_neighbors(graph, nid, n)
            n_hop_of_GP |= neighbors

    return n_hop_of_GP


def main(cfg):
    dataset_name = cfg.dataset.name

    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [str(nid) for nid in GP_nids]
    print(f"There are {len(GPs)} malicious nodes")

    neighbors = set()

    for attack in dataset_to_time[dataset_name]:
        start_time, end_time = attack[0], attack[1]

        print("Generating graph")
        graph = gen_graph(datetime_to_ns_time_US(start_time), datetime_to_ns_time_US(end_time), cfg)

        n = 2
        print(f"Get {n}-hop neighbors of GPs")
        neigh = get_n_hop_of_GP(graph, GPs, n)

        neighbors |= neigh

    print(f"In {dataset_name}: {len(neigh)}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
