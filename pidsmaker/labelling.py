import csv
import os.path
from collections import defaultdict

from provnet_utils import init_database_connection, log, datetime_to_ns_time_US

def get_ground_truth(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    for file in cfg.dataset.ground_truth_relative_path:
        with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
                uuid_to_node_id[node_uuid] = str(node_id)

    mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        num_GPs= len(ground_truth_nids)
        for file in cfg.dataset.ground_truth_relative_path:
            file_name = file.split('/')[-1]
            with open(os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, file_name), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    ground_truth_nids.append(int(node_id))
                    ground_truth_paths[int(node_id)] = node_labels
                    uuid_to_node_id[node_uuid] = str(node_id)
        num_mimicry_GPs = len(ground_truth_nids) - num_GPs
        log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")

    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id

def get_GP_of_each_attack(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_to_nids = {}

    for i, (path, attack_to_time_window) in enumerate(zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)):
        attack_to_nids[i] = {}
        attack_to_nids[i]["nids"] = set()
        attack_to_nids[i]["time_range"] =[datetime_to_ns_time_US(tw) for tw in [attack_to_time_window[1], attack_to_time_window[2]]]
        
        with open(os.path.join(cfg._ground_truth_dir, path), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                attack_to_nids[i]["nids"].add(int(node_id))

        mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_mimicry_GPs = 0
            with open(os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, path.split('/')[-1]), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    num_mimicry_GPs += 1
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    attack_to_nids[i]["nids"].add(int(node_id))
            log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")
    return attack_to_nids

def get_uuid2nids(cur):
    queries = {
        "file": "SELECT index_id, node_uuid FROM file_node_table;",
        "netflow": "SELECT index_id, node_uuid FROM netflow_node_table;",
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;"
    }
    uuid2nids = {}
    nid2uuid = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            uuid2nids[row[1]] = row[0]
            nid2uuid[row[0]] = row[1]

    return uuid2nids, nid2uuid

def get_events(cur,
               start_time,
               end_time,):
    # malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    # sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str});"
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"

    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_t2malicious_node(cfg) -> dict[list]:
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        ground_truth_nids = set()
        with open(os.path.join(cfg._ground_truth_dir, attack), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.add(str(node_id))

        mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_GPs= len(ground_truth_nids)
            with open(os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, attack.split('/')[-1]), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    ground_truth_nids.add(str(node_id))
            num_mimicry_GPs = len(ground_truth_nids) - num_GPs
            log(f"{num_mimicry_GPs} mimicry nodes loaded")

        rows = get_events(cur, start_time, end_time)
        for row in rows:
            src_id = row[1]
            dst_id = row[4]
            t = row[6]
            if src_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(src_id)])
            if dst_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    return t_to_node

def get_attack_to_mal_edges(cfg) -> dict[list]:
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    malicious_edge_selection = cfg.detection.evaluation.edge_evaluation.malicious_edge_selection
    
    attack_to_mal_edges = defaultdict(set)
    for i, (path, attack_to_time_window) in enumerate(zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)):

        start_time = datetime_to_ns_time_US(attack_to_time_window[1])
        end_time = datetime_to_ns_time_US(attack_to_time_window[2])
        
        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, path), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))
        ground_truth_nids = set(ground_truth_nids)

        rows = get_events(cur, start_time, end_time)
        for row in rows:
            src_idx_id = row[1]
            ope = row[2]
            dst_idx_id = row[4]
            event_uuid = row[5]
            timestamp_rec = row[6]

            condition = None
            if malicious_edge_selection == "src_node":
                condition = src_idx_id in ground_truth_nids
            elif malicious_edge_selection == "dst_node":
                condition = dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "both_nodes":
                condition = src_idx_id in ground_truth_nids and dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "either_node":
                condition = src_idx_id in ground_truth_nids or dst_idx_id in ground_truth_nids
            else:
                raise ValueError("`malicious_edge_selection` must be one of 'src_node', 'dst_node', 'both_nodes', 'either_node")
            
            if condition:
                attack_to_mal_edges[i].add((src_idx_id, dst_idx_id, timestamp_rec, ope))
    
    return attack_to_mal_edges

def get_ground_truth_edges(cfg) -> set:
    attack_to_mal_edges = get_attack_to_mal_edges(cfg)

    malicious_edges = set()
    for attack, edges_set in attack_to_mal_edges.items():
        malicious_edges |= edges_set

    return malicious_edges
