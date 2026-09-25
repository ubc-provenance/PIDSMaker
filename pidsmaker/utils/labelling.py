import csv
import os.path
from collections import defaultdict

from pidsmaker.utils.utils import datetime_to_ns_time_US, init_database_connection, log


def is_threatrace(cfg):
    """True when the ThreaTrace ground truth is selected."""
    return cfg.evaluation.ground_truth_version == "threatrace"


def _threatrace_gt_file(cfg):
    """ThreaTrace ships one flat UUID list per dataset. Reuse the dataset's
    ground-truth subfolder (e.g. 'E3-CADETS') from its configured paths."""
    paths = cfg.dataset.ground_truth_relative_path
    if not paths:
        raise ValueError(
            "ThreaTrace ground truth needs the dataset's GT subfolder, but "
            "`ground_truth_relative_path` is empty for this dataset."
        )
    return os.path.join(paths[0].split("/")[0], "ground_truth.txt")


def ensure_ground_truth_available(cfg):
    """Fail fast with a clear message when the selected ground truth has no file for
    this dataset. ThreaTrace only covers CADETS, THEIA, TRACE, and FIVEDIRECTIONS (E3)."""
    if not is_threatrace(cfg):
        return
    paths = cfg.dataset.ground_truth_relative_path
    subdir = paths[0].split("/")[0] if paths else None
    gt_path = os.path.join(cfg._ground_truth_dir, subdir, "ground_truth.txt") if subdir else None
    if not gt_path or not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"ThreaTrace ground truth not found for dataset '{cfg.dataset.name}'. "
            f"ThreaTrace provides ground truth only for CADETS, THEIA, TRACE, and "
            f"FIVEDIRECTIONS (E3)."
        )


def _test_period_ns(cfg):
    """(start, end) of the dataset's test period in ns — the single combined
    window used for the flat ThreaTrace ground truth."""
    dates = sorted(cfg.dataset.test_dates)
    return (
        datetime_to_ns_time_US(dates[0] + " 00:00:00"),
        datetime_to_ns_time_US(dates[-1] + " 23:59:59"),
    )


def ground_truth_files(cfg):
    """GT file relative paths for the active version: one flat file for
    ThreaTrace, the per-attack files otherwise."""
    if is_threatrace(cfg):
        return [_threatrace_gt_file(cfg)]
    return list(cfg.dataset.ground_truth_relative_path)


def ground_truth_attacks(cfg):
    """(relative_path, start_ns, end_ns) per attack. ThreaTrace collapses to a
    single combined window over the dataset's test period."""
    if is_threatrace(cfg):
        start, end = _test_period_ns(cfg)
        return [(_threatrace_gt_file(cfg), start, end)]
    return [
        (a[0], datetime_to_ns_time_US(a[1]), datetime_to_ns_time_US(a[2]))
        for a in cfg.dataset.attack_to_time_window
    ]


def read_ground_truth_uuids(cfg, relative_path):
    """Yield (uuid, label) from a GT file, handling both the 3-column
    orthrus/reapr CSV and the UUID-only ThreaTrace txt."""
    with open(os.path.join(cfg._ground_truth_dir, relative_path), "r") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            yield row[0].strip(), (row[1] if len(row) > 1 else "threatrace")


def _mimicry_nodes(cfg, relative_path, uuid2nids):
    """Node ids injected by the mimicry attack generator for one ground-truth file.
    Only applies to the per-attack (orthrus/reapr) ground truth."""
    nodes = {}
    path = os.path.join(cfg.construction._mimicry_dir, relative_path.split("/")[-1])
    with open(path, "r") as f:
        for row in csv.reader(f):
            node_id = uuid2nids.get(row[0])
            if node_id is not None:
                nodes[int(node_id)] = row[1] if len(row) > 1 else ""
    return nodes


def get_ground_truth(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    missing = 0
    for file in ground_truth_files(cfg):
        for node_uuid, node_labels in read_ground_truth_uuids(cfg, file):
            node_id = uuid2nids.get(node_uuid)
            if node_id is None:
                missing += 1
                continue
            ground_truth_nids.append(int(node_id))
            ground_truth_paths[int(node_id)] = node_labels
            uuid_to_node_id[node_uuid] = str(node_id)
    if missing:
        log(f"{missing} ground-truth UUIDs not present in the graph (skipped)")

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num and mimicry_edge_num > 0 and not is_threatrace(cfg):
        num_before = len(ground_truth_nids)
        for file in cfg.dataset.ground_truth_relative_path:
            for node_id, labels in _mimicry_nodes(cfg, file, uuid2nids).items():
                ground_truth_nids.append(node_id)
                ground_truth_paths[node_id] = labels
        log(f"{len(ground_truth_nids) - num_before} mimicry ground truth nodes loaded")

    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id


def get_GP_of_each_attack(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_to_nids = {}
    for i, (path, start, end) in enumerate(ground_truth_attacks(cfg)):
        nids = set()
        for node_uuid, _labels in read_ground_truth_uuids(cfg, path):
            node_id = uuid2nids.get(node_uuid)
            if node_id is not None:
                nids.add(int(node_id))

        mimicry_edge_num = cfg.construction.mimicry_edge_num
        if mimicry_edge_num and mimicry_edge_num > 0 and not is_threatrace(cfg):
            mimicry = _mimicry_nodes(cfg, path, uuid2nids)
            nids |= set(mimicry.keys())
            log(f"{len(mimicry)} mimicry ground truth nodes loaded")

        attack_to_nids[i] = {"nids": nids, "time_range": [start, end]}
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
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"

    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def get_t2malicious_node(cfg) -> dict[list]:
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    for path, start_time, end_time in ground_truth_attacks(cfg):
        ground_truth_nids = set()
        for node_uuid, _labels in read_ground_truth_uuids(cfg, path):
            node_id = uuid2nids.get(node_uuid)
            if node_id is not None:
                ground_truth_nids.add(str(node_id))

        mimicry_edge_num = cfg.construction.mimicry_edge_num
        if mimicry_edge_num and mimicry_edge_num > 0 and not is_threatrace(cfg):
            mimicry = _mimicry_nodes(cfg, path, uuid2nids)
            ground_truth_nids |= {str(n) for n in mimicry}
            log(f"{len(mimicry)} mimicry nodes loaded")

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

    malicious_edge_selection = cfg.evaluation.edge_evaluation.malicious_edge_selection

    attack_to_mal_edges = defaultdict(set)
    for i, (path, start_time, end_time) in enumerate(ground_truth_attacks(cfg)):
        ground_truth_nids = set()
        for node_uuid, _labels in read_ground_truth_uuids(cfg, path):
            node_id = uuid2nids.get(node_uuid)
            if node_id is not None:
                ground_truth_nids.add(str(node_id))

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
                raise ValueError(
                    "`malicious_edge_selection` must be one of 'src_node', 'dst_node', 'both_nodes', 'either_node"
                )

            if condition:
                attack_to_mal_edges[i].add((src_idx_id, dst_idx_id, timestamp_rec, ope))

    return attack_to_mal_edges


def get_ground_truth_edges(cfg) -> set:
    attack_to_mal_edges = get_attack_to_mal_edges(cfg)

    malicious_edges = set()
    for attack, edges_set in attack_to_mal_edges.items():
        malicious_edges |= edges_set

    return malicious_edges
