import csv
import os.path
import time
from collections import defaultdict
from datetime import datetime, timezone

import pytz

from pidsmaker.utils.utils import datetime_to_ns_time_US, init_database_connection, log


def _datetime_to_ns(date_str, cfg):
    """Convert a datetime string to nanoseconds, respecting the dataset timezone.

    Datasets with timezone='UTC' use UTC.
    All other datasets default to US/Eastern (DARPA TC convention).
    """
    tz_name = getattr(cfg.dataset, "timezone", None)
    if tz_name and tz_name.upper() == "UTC":
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    return datetime_to_ns_time_US(date_str)


def get_ground_truth(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    for file in cfg.dataset.ground_truth_relative_path:
        with open(os.path.join(cfg._ground_truth_dir, file), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
                uuid_to_node_id[node_uuid] = str(node_id)

    mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        num_GPs = len(ground_truth_nids)
        for file in cfg.dataset.ground_truth_relative_path:
            file_name = file.split("/")[-1]
            with open(
                os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, file_name), "r"
            ) as f:
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

    for i, (path, attack_to_time_window) in enumerate(
        zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)
    ):
        attack_to_nids[i] = {}
        attack_to_nids[i]["nids"] = set()
        attack_to_nids[i]["time_range"] = [
            _datetime_to_ns(tw, cfg)
            for tw in [attack_to_time_window[1], attack_to_time_window[2]]
        ]

        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                attack_to_nids[i]["nids"].add(int(node_id))

        mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_mimicry_GPs = 0
            with open(
                os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, path.split("/")[-1]), "r"
            ) as f:
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
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;",
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


def get_events(
    cur,
    start_time,
    end_time,
):
    # malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    # sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str});"
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"

    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def get_t2malicious_node(cfg) -> dict[list]:
    """Map event timestamps to malicious node UUIDs.

    When edge-level ground truth CSVs are available, only events
    matching explicit attack edges are considered malicious.  Otherwise falls back
    to the DARPA TC approach: any event touching a GT node within the attack
    window is malicious.
    """
    if _has_edge_csvs(cfg):
        return _get_t2malicious_node_from_edge_gt(cfg)
    return _get_t2malicious_node_from_node_gt(cfg)


def _get_t2malicious_node_from_edge_gt(cfg) -> dict[list]:
    """Use explicit edge-level ground truth to find malicious events.

    For each attack edge (src, dst, event_type), query the DB for matching events
    and record the timestamp -> node mapping.
    """
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    edge_paths = cfg.dataset.ground_truth_edges_relative_path
    for path in edge_paths:
        filepath = os.path.join(cfg._ground_truth_dir, path)
        if not os.path.exists(filepath):
            continue

        # Collect all attack edges: (src_nid, dst_nid, event_type)
        attack_edges = set()
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_nid = uuid2nids.get(row["src_node_uuid"])
                dst_nid = uuid2nids.get(row["dst_node_uuid"])
                if src_nid is not None and dst_nid is not None:
                    attack_edges.add((str(src_nid), str(dst_nid), row["event_type"]))

        if not attack_edges:
            continue

        # Query events for each attack edge
        for src_nid, dst_nid, event_type in attack_edges:
            sql = (
                "SELECT src_index_id, dst_index_id, timestamp_rec "
                "FROM event_table "
                "WHERE src_index_id = %s AND dst_index_id = %s AND operation = %s;"
            )
            cur.execute(sql, (src_nid, dst_nid, event_type))
            for row in cur.fetchall():
                t = int(row[2])
                t_to_node[t].append(nid2uuid[int(row[0])])
                t_to_node[t].append(nid2uuid[int(row[1])])

    log(f"Edge GT: found {len(t_to_node)} event timestamps with malicious activity")
    return t_to_node


def _get_t2malicious_node_from_node_gt(cfg) -> dict[list]:
    """Original DARPA TC approach: any event touching a GT node in the attack window."""
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = _datetime_to_ns(attack_tuple[1], cfg)
        end_time = _datetime_to_ns(attack_tuple[2], cfg)

        ground_truth_nids = set()
        with open(os.path.join(cfg._ground_truth_dir, attack), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.add(str(node_id))

        mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_GPs = len(ground_truth_nids)
            with open(
                os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, attack.split("/")[-1]),
                "r",
            ) as f:
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


def _has_edge_csvs(cfg):
    """Check if explicit edge-level ground truth CSVs are configured."""
    paths = getattr(cfg.dataset, "ground_truth_edges_relative_path", None)
    return paths is not None and len(paths) > 0


def get_attack_to_mal_edges(cfg) -> dict[set]:
    """Load per-attack malicious edges.

    If ground_truth_edges_relative_path is configured,
    loads explicit attack edges from CSV files.
    Otherwise falls back to the DB-query approach (DARPA TC datasets).
    """
    if _has_edge_csvs(cfg):
        return _get_attack_to_mal_edges_from_csv(cfg)
    return _get_attack_to_mal_edges_from_db(cfg)


def _get_attack_to_mal_edges_from_csv(cfg) -> dict[set]:
    """Load malicious edges from explicit CSV files.

    CSV format: src_node_uuid,dst_node_uuid,event_type,attack_edge_type
    Edges are matched by (src_index_id, dst_index_id, event_type) — no timestamp needed.
    """
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_to_mal_edges = defaultdict(set)
    edge_paths = cfg.dataset.ground_truth_edges_relative_path

    for i, path in enumerate(edge_paths):
        filepath = os.path.join(cfg._ground_truth_dir, path)
        if not os.path.exists(filepath):
            log(f"Warning: edge ground truth file not found: {filepath}")
            continue
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_uuid = row["src_node_uuid"]
                dst_uuid = row["dst_node_uuid"]
                event_type = row["event_type"]

                src_nid = uuid2nids.get(src_uuid)
                dst_nid = uuid2nids.get(dst_uuid)
                if src_nid is None or dst_nid is None:
                    continue

                # Store as (src_index_id, dst_index_id, event_type) — no timestamp
                attack_to_mal_edges[i].add((str(src_nid), str(dst_nid), event_type))

    total = sum(len(s) for s in attack_to_mal_edges.values())
    log(f"Loaded {total} malicious edges from {len(edge_paths)} CSV files")
    return attack_to_mal_edges


def _get_attack_to_mal_edges_from_db(cfg) -> dict[set]:
    """Original DB-query approach: find all events involving GT nodes during attack windows."""
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    malicious_edge_selection = cfg.detection.evaluation.edge_evaluation.malicious_edge_selection

    attack_to_mal_edges = defaultdict(set)
    for i, (path, attack_to_time_window) in enumerate(
        zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)
    ):
        start_time = _datetime_to_ns(attack_to_time_window[1], cfg)
        end_time = _datetime_to_ns(attack_to_time_window[2], cfg)

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
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
                raise ValueError(
                    "`malicious_edge_selection` must be one of 'src_node', 'dst_node', 'both_nodes', 'either_node"
                )

            if condition:
                attack_to_mal_edges[i].add((src_idx_id, dst_idx_id, timestamp_rec, ope))

    return attack_to_mal_edges


def get_ground_truth_edges(cfg) -> set:
    """Return the set of all malicious edges across all attacks.

    When using CSV-based edge labels, edges are (src_nid, dst_nid, event_type) tuples.
    When using DB-based labels, edges are (src_nid, dst_nid, timestamp, event_type) tuples.
    """
    attack_to_mal_edges = get_attack_to_mal_edges(cfg)

    malicious_edges = set()
    for attack, edges_set in attack_to_mal_edges.items():
        malicious_edges |= edges_set

    return malicious_edges
