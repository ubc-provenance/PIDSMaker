from collections import defaultdict

from config import *
from provnet_utils import *

def get_ground_truth(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

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
    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id

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

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def get_events(cur,
               start_time,
               end_time,
               malicious_nodes : list):
    malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str});"
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

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))

            rows = get_events(cur, start_time, end_time, ground_truth_nids)
            for row in rows:
                src_id = row[1]
                dst_id = row[4]
                t = row[6]
                if src_id in ground_truth_nids:
                    t_to_node[int(t)].append(nid2uuid[int(src_id)])
                if dst_id in ground_truth_nids:
                    t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    return t_to_node
