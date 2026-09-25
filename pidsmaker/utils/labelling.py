import csv
import os.path
from collections import defaultdict

from pidsmaker.utils.utils import datetime_to_ns_time_US, init_database_connection, log
from pidsmaker.utils.dataset_utils import get_uuid_to_index_id, get_time_and_end_node


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
        datetime_to_ns_time_US(dates[0] + " 00:00:00", timezone=cfg.dataset.timezone),
        datetime_to_ns_time_US(dates[-1] + " 23:59:59", timezone=cfg.dataset.timezone),
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
        (a[0], datetime_to_ns_time_US(a[1], timezone=cfg.dataset.timezone), datetime_to_ns_time_US(a[2], timezone=cfg.dataset.timezone))
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
    uuid2nids, nid2uuid = get_uuid_to_index_id(cfg)

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
    uuid2nids, _ = get_uuid_to_index_id(cfg)

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
    uuid2nids, nid2uuid = get_uuid_to_index_id(cfg)

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
    uuid2nids, nid2uuid = get_uuid_to_index_id(cfg)

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

        rows = get_time_and_end_node(cfg, start_time, end_time)
        for row in rows:
            src_id = str(row[0])
            dst_id = str(row[1])
            t = row[2]
            if src_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(src_id)])
            if dst_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    log(f"Node GT: found {len(t_to_node)} event timestamps with malicious activity")
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
    uuid2nids, _ = get_uuid_to_index_id(cfg)

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
    uuid2nids, nid2uuid = get_uuid_to_index_id(cfg)

    malicious_edge_selection = cfg.evaluation.edge_evaluation.malicious_edge_selection

    attack_to_mal_edges = defaultdict(set)
    for i, (path, start_time, end_time) in enumerate(ground_truth_attacks(cfg)):
        ground_truth_nids = set()
        for node_uuid, _labels in read_ground_truth_uuids(cfg, path):
            node_id = uuid2nids.get(node_uuid)
            if node_id is not None:
                ground_truth_nids.add(str(node_id))

        rows = get_time_and_end_node(cfg, start_time, end_time)
        for row in rows:
            src_idx_id = str(row[0])
            dst_idx_id = str(row[1])
            timestamp_rec = row[2]
            ope = row[3]

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
