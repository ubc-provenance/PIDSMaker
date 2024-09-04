from provnet_utils import *
from config import *

import numpy as np
from labelling import get_ground_truth
from detection.evaluation_methods.evaluation_utils import compute_tw_labels
import networkx as nx
import datetime
from labelling import datetime_to_ns_time_US,get_uuid2nids

def get_events_between_GPs(cur,
               start_time,
               end_time,
               malicious_nodes : list):
    malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str}) AND dst_index_id IN ({malicious_nodes_str});"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def get_events_between_time_range(cur,
               start_time,
               end_time,):
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows
def generate_DAG(edges):
    node_version = {}
    for (u, v, t) in edges:
        if u not in node_version:
            node_version[u] = 0
        if v not in node_version:
            node_version[v] = 0

    sorted_edges = sorted(edges, key=lambda x: x[2])

    new_nodes = set()
    new_edges = []
    visited = set()
    for u, v, t in sorted_edges:

        if u == v:
            continue

        src = str(u) + '-' + str(node_version[u])
        visited.add(u)
        new_nodes.add(src)

        if v not in visited:
            dst = str(v) + '-' + str(node_version[v])
            visited.add(v)
            new_nodes.add(dst)
            new_edges.append((src, dst, {'time': int(t)}))
        else:
            dst_current = str(v) + '-' + str(node_version[v])
            dst_new = str(v) + '-' + str(node_version[v] + 1)
            node_version[v] += 1
            new_nodes.add(dst_new)
            new_edges.append((src, dst_new, {'time': int(t)}))
            new_edges.append((dst_current, dst_new, {'time': int(t)}))

    DAG = nx.DiGraph()
    DAG.add_nodes_from(list(new_nodes))
    DAG.add_edges_from(new_edges)

    return DAG, node_version

def main(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_GPs = {}
    attack_to_des = {}
    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        print("==" * 30)
        print(f"start processing attak {attack}")

        print("get GPs")
        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))

        attack_GPs[attack] = set(ground_truth_nids)

        print("Get events between GPs")
        rows = get_events_between_GPs(cur, start_time, end_time, ground_truth_nids)

        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id),str(dst_id),int(t)))

        dag_between_GPs, _ = generate_DAG(edges)

        print("Get root nodes")
        root_nodes = set([node for node, in_degree in dag_between_GPs.in_degree() if in_degree == 0])

        print(f"Root nodes are: {root_nodes}")

        print("Get events in attack time range")
        rows = get_events_between_time_range(cur, start_time, end_time)
        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_of_attack, node_version = generate_DAG(edges)

        all_descendants = set()
        for root in root_nodes:
            descendants = nx.descendants(dag_of_attack, root)
            desc = set([v.split('-')[0] for v in descendants])
            all_descendants |= desc

        print(f"{len(all_descendants)} descedants of root nodes in the attack")

        attack_to_des[attack] = all_descendants

        print("==" * 30)

    root_desc = set()
    print("The number of descendants of root nodes:")
    for attack, desc_set in attack_to_des.items():
        print(f"{attack} : {len(desc_set)}")
        root_desc |= desc_set

    print(f"Total : {len(root_desc)}")

    pass

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)