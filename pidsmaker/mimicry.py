import csv
import os
import random
from collections import defaultdict

import networkx as nx

from pidsmaker.config import *
from pidsmaker.dataset_utils import get_rel2id
from pidsmaker.provnet_utils import *
from pidsmaker.provnet_utils import datetime_to_ns_time_US

possible_structure = [
    ["EVENT_OPEN", "EVENT_READ"],
    ["EVENT_OPEN", "EVENT_WRITE"],
    ["EVENT_OPEN", "EVENT_READ", "EVENT_WRITE"],
    ["EVENT_OPEN", "EVENT_WRITE", "EVENT_READ"],
    ["EVENT_OPEN", "EVENT_READ", "EVENT_WRITE", "EVENT_READ", "EVENT_WRITE"],
    ["EVENT_OPEN", "EVENT_WRITE", "EVENT_READ", "EVENT_WRITE", "EVENT_READ"],
    ["EVENT_OPEN", "EVENT_READ", "EVENT_WRITE", "EVENT_WRITE", "EVENT_READ"],
    ["EVENT_OPEN", "EVENT_WRITE", "EVENT_READ", "EVENT_WRITE", "EVENT_WRITE"],
    [
        "EVENT_OPEN",
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_WRITE",
    ],
    [
        "EVENT_OPEN",
        "EVENT_WRITE",
        "EVENT_READ",
        "EVENT_READ",
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_READ",
    ],
    [
        "EVENT_OPEN",
        "EVENT_READ",
        "EVENT_READ",
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_READ",
    ],
    [
        "EVENT_OPEN",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_WRITE",
    ],
    ["EVENT_OPEN", "EVENT_READ", "EVENT_READ", "EVENT_READ", "EVENT_READ"],
    [
        "EVENT_OPEN",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_WRITE",
        "EVENT_WRITE",
    ],
]


def get_uuid2nids2type(cur):
    queries = {
        "file": "SELECT index_id, node_uuid FROM file_node_table;",
        "netflow": "SELECT index_id, node_uuid FROM netflow_node_table;",
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;",
    }
    uuid2nids = {}
    nid2uuid = {}
    nid2type = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            uuid2nids[row[1]] = row[0]
            nid2uuid[row[0]] = row[1]
            nid2type[row[0]] = node_type
    return uuid2nids, nid2uuid, nid2type


def obtain_all_files(cur):
    sql = "SELECT index_id, node_uuid FROM file_node_table;"
    cur.execute(sql)
    rows = cur.fetchall()

    file_set = set()
    for row in rows:
        file_set.add(row[0])
    return file_set


def divide_integer(a, b):
    base_size = a // b
    remainder = a % b
    result = [base_size + 1 if i < remainder else base_size for i in range(b)]
    return result


def random_timestamp(start_ns, end_ns):
    if start_ns > end_ns:
        start_ns, end_ns = end_ns, start_ns
    return random.randint(start_ns, end_ns)


def save_mimicry_nodes(cfg, cur, nodes, attack):
    os.makedirs(cfg.preprocessing.build_graphs._mimicry_dir, exist_ok=True)
    save_dir = os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, attack.split("/")[-1])

    data = []
    sql1 = "SELECT node_uuid, index_id, path FROM file_node_table;"
    sql2 = "SELECT node_uuid, index_id, path, cmd FROM subject_node_table;"

    cur.execute(sql1)
    rows = cur.fetchall()
    for row in rows:
        node_uuid = row[0]
        index_id = int(row[1])
        path = row[2] if row[2] is not None else "None"
        if index_id in nodes:
            data.append((node_uuid, {"file": path}, index_id))

    cur.execute(sql2)
    rows = cur.fetchall()
    for row in rows:
        node_uuid = row[0]
        index_id = int(row[1])
        path = row[2] if row[2] is not None else "None"
        cmd = row[3] if row[3] is not None else "None"
        if index_id in nodes:
            data.append((node_uuid, {"subject": path + " " + cmd}, index_id))

    with open(save_dir, "w") as f:
        csv_writer = csv.writer(f)
        for line in data:
            csv_writer.writerow(line)

    log(f"{len(data)} mimicry nodes saved into {save_dir}.")


def gen_mimicry_edges(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid, nid2type = get_uuid2nids2type(cur)

    rel2id = get_rel2id(cfg)

    mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
    attack_num = len(cfg.dataset.ground_truth_relative_path)
    num_each_att = divide_integer(mimicry_edge_num, attack_num)

    attack_index = 0
    attack_GPs = {}
    attack_mimicry_events = {}
    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        # Obtain mimicry-connected nodes as new malicious nodes
        mimicry_GPs = set()

        # Find attack root and descendant process
        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))

        attack_GPs[attack] = set(ground_truth_nids)

        # Create graph and obtain root nodes
        rows = get_events_between_GPs(cur, start_time, end_time, ground_truth_nids)

        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = int(row[6])
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_between_GPs, _ = generate_DAG(edges)
        root_nodes = set(
            [node for node, in_degree in dag_between_GPs.in_degree() if in_degree == 0]
        )

        # create graph and obtain all descendants
        rows = get_events_between_time_range(cur, start_time, end_time)
        edges = []
        process2times = defaultdict(list)
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = int(row[6])
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

            if nid2type[int(src_id)] == "subject":
                process2times[src_id].append(t)
            if nid2type[int(dst_id)] == "subject":
                process2times[dst_id].append(t)

        dag_of_attack, node_version = generate_DAG(edges)

        all_descendants = set()
        for root in root_nodes:
            descendants = nx.descendants(dag_of_attack, root)
            desc = set([v.split("-")[0] for v in descendants])
            all_descendants |= desc
        all_descendants |= set([v.split("-")[0] for v in root_nodes])

        possible_process = set()
        for nid in all_descendants:
            if nid2type[int(nid)] == "subject":
                possible_process.add(nid)

        possible_file = obtain_all_files(cur)

        possible_process = list(possible_process)
        possible_file = list(possible_file)

        # Generate mimicry events
        mimicry_events = []
        events_needed = num_each_att[attack_index]
        while len(mimicry_events) < events_needed:
            random_subject = random.choice(possible_process)
            random_file = random.choice(possible_file)
            random_time = random.choice(process2times[random_subject])
            random_structure = random.choice(possible_structure)

            mimicry_GPs.add(int(random_subject))
            mimicry_GPs.add(int(random_file))

            current_time = int(random_time)

            for ope in random_structure:
                time_bias = random.randint(0, int(3 * 1e9))
                event_time = current_time + time_bias
                current_time = event_time
                event_uuid = stringtomd5(str(random_subject) + str(random_file) + str(current_time))
                if ope == "EVENT_OPEN" or ope == "EVENT_WRITE":
                    event = (
                        nid2uuid[int(random_subject)],
                        str(random_subject),
                        ope,
                        nid2uuid[int(random_file)],
                        str(random_file),
                        event_uuid,
                        event_time,
                        0,
                    )
                elif ope == "EVENT_READ":
                    event = (
                        nid2uuid[int(random_file)],
                        str(random_file),
                        ope,
                        nid2uuid[int(random_subject)],
                        str(random_subject),
                        event_uuid,
                        event_time,
                        0,
                    )
                else:
                    log(f"Undefined event type {ope}")

                mimicry_events.append(event)

        save_mimicry_nodes(cfg, cur, mimicry_GPs, attack)

        attack_mimicry_events[attack_index] = mimicry_events
        log(f"For attack {attack_index}, {len(mimicry_events)} mimicry events are generated")
        attack_index += 1

    return attack_mimicry_events
