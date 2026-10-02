"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation.
"""

import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.utils.dataset_utils import get_rel2id, rel2id_optc, OPTC_DATASETS, _has_consistent_edge_types, translate_optc_to_tc
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    get_split_to_files,
    init_database_connection,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    stringtomd5,
)


_cursor_id = 0


def _next_cursor_name():
    global _cursor_id
    _cursor_id += 1
    return f"pids_cur_{_cursor_id}"


def _stream_rows(connect, sql, params=None, fetch_size=50_000):
    """Execute a query via a server-side named cursor, yielding rows incrementally.

    psycopg2 regular cursors buffer the entire result set on the client after
    execute(). Named cursors keep results on the server and fetch in small
    batches, keeping client-side memory bounded regardless of result size.
    """
    cur = connect.cursor(_next_cursor_name())
    try:
        cur.execute(sql, params)
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            yield from rows
    finally:
        cur.close()



def compute_indexid2msg(cfg):
    """Compute mapping from node index IDs to node types and feature labels.

    Queries PostgreSQL database for all nodes (netflow, subject/process, file) and
    extracts their attributes to create feature labels based on configuration.

    Args:
        cfg: Configuration with database connection and feature settings

    Returns:
        dict: Mapping {index_id: [node_type, label_string]} where:
            - index_id: Database node identifier
            - node_type: One of 'netflow', 'subject', 'file'
            - label_string: Feature label (hashed or plaintext depending on config)
    """
    _, connect = init_database_connection(cfg)

    use_hashed_label = cfg.construction.use_hashed_label
    null_label_tokens = cfg.construction.null_label_tokens
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    indexid2msg = {}

    _NONE_VALUES = {"None", "NA", "0", "none", "null", ""}

    def _auto_label_default(attrs):
        """Default auto-label: concatenate all non-null, non-type attributes."""
        parts = []
        for key in sorted(attrs):
            if key == "type":
                continue
            val = attrs[key]
            if val not in _NONE_VALUES:
                parts.append(val)
        return " ".join(parts)

    def get_label_str_from_features(attrs, node_type):
        """Extract feature label from node attributes based on configured features.

        Args:
            attrs: Dictionary of node attributes
            node_type: Type of node ('netflow', 'subject', 'file')

        Returns:
            str: Space-separated feature string, optionally hashed
        """
        feats = node_label_features[node_type]
        if feats == ["auto"]:
            if node_type == "subject":
                # Prefer cmd_line; fall back to path if cmd_line is null
                cmd_line = attrs.get("cmd_line", "")
                path = attrs.get("path", "")
                if cmd_line not in _NONE_VALUES:
                    label_str = cmd_line
                elif path not in _NONE_VALUES:
                    label_str = ("[NO_CMD] " + path) if null_label_tokens else path
                else:
                    label_str = "[NO_CMD]" if null_label_tokens else ""
            elif node_type == "file":
                path = attrs.get("path", "")
                if path not in _NONE_VALUES:
                    label_str = path
                else:
                    label_str = "[NO_PATH]" if null_label_tokens else ""
            elif node_type == "netflow":
                label_str = _auto_label_default(attrs)
                if not label_str and null_label_tokens:
                    label_str = "[NO_IP]"
            else:
                label_str = _auto_label_default(attrs)
        else:
            label_str = " ".join([attrs[label_used] for label_used in feats])

        if use_hashed_label:
            label_str = stringtomd5(label_str)
        # Keep only printable ASCII — provenance labels are paths, cmds, IPs
        label_str = re.sub(r"[^\x20-\x7e]+", "", label_str).strip()
        return label_str

    # netflow
    netflow_count = 0
    for i in _stream_rows(connect, "select * from netflow_node_table;"):
        attrs = {
            "type": "netflow",
            "local_ip": str(i[2]),
            "local_port": str(i[3]),
            "remote_ip": str(i[4]),
            "remote_port": str(i[5]),
        }
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)
        indexid2msg[index_id] = [node_type, label_str]
        netflow_count += 1

    log(f"Number of netflow nodes: {netflow_count}")

    # subject
    subject_count = 0
    for i in _stream_rows(connect, "select * from subject_node_table;"):
        attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)
        indexid2msg[index_id] = [node_type, label_str]
        subject_count += 1

    log(f"Number of process nodes: {subject_count}")

    # file
    file_count = 0
    for i in _stream_rows(connect, "select * from file_node_table;"):
        attrs = {"type": "file", "path": str(i[2])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)
        indexid2msg[index_id] = [node_type, label_str]
        file_count += 1

    log(f"Number of file nodes: {file_count}")

    return indexid2msg  # {index_id: [node_type, msg]}


def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """Save filtered node index-to-feature mapping to disk.

    Filters out nodes not used in any train/val/test graphs (due to excluded edge types)
    before saving to avoid downstream errors during featurization.

    Note: Must be called after graph construction to ensure only used nodes are saved.

    Args:
        indexid2msg: Full node mapping from compute_indexid2msg()
        split2nodes: Mapping of splits to their node sets
        cfg: Configuration with output directory path
    """
    all_nodes = set().union(*(split2nodes[split] for split in ["train", "val", "test"]))
    indexid2msg = {k: v for k, v in indexid2msg.items() if k in all_nodes}

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg):
    """Compute and save mapping of dataset splits to their node sets.

    Loads all graphs from train/val/test splits and collects unique node IDs
    appearing in each split. Used to filter node features and track split membership.

    Args:
        cfg: Configuration with graph directory and split file paths

    Returns:
        dict: Mapping of split names to node sets:
            {'train': {node_ids}, 'val': {node_ids}, 'test': {node_ids}}
    """
    split_to_files = get_split_to_files(cfg, cfg.construction._graphs_dir)
    split2nodes = defaultdict(set)

    for split, files in split_to_files.items():
        for path in log_tqdm(files, desc=f"Check nodes in {split} set"):
            G = torch.load(path)
            for node in G.nodes():
                split2nodes[split].add(node)
    split2nodes = dict(split2nodes)

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


def gen_edge_fused_tw(indexid2msg, cfg):
    """Generate time-windowed provenance graphs from database events.

    Main graph construction function that:
    1. Queries database for events in time windows
    2. Optionally fuses consecutive edges of same type between node pairs
    3. Optionally adds attack mimicry events for data augmentation
    4. Builds NetworkX MultiDiGraphs with node attributes and edge metadata
    5. Saves graphs to disk organized by day and time window

    Args:
        indexid2msg: Node index to [type, label] mapping from compute_indexid2msg()
        cfg: Configuration with:
            - Database connection settings
            - Time window parameters (size, dates)
            - Edge type filtering (rel2id)
            - Mimicry settings (mimicry_edge_num)
            - Output directory paths
    """
    _, connect = init_database_connection(cfg)
    rel2id = get_rel2id(cfg)
    # DB rows carry raw edge type strings (e.g. "START", "MESSAGE" for OpTC).
    # When consistent_edge_types is on, get_rel2id returns TC-equivalent labels,
    # so we must filter against the raw OpTC vocabulary instead of the translated one.
    include_edge_type = rel2id_optc if cfg.dataset.name in OPTC_DATASETS else rel2id

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        attack_mimicry_events = mimicry.gen_mimicry_edges(cfg)
    else:
        attack_mimicry_events = defaultdict(list)

    # In test mode, we ensure to get 1 TW in each set
    dates = get_dates_from_cfg(cfg)

    # Timezone-aware timestamp conversion: UTC for datasets with timezone=UTC, else US/Eastern
    _tz = getattr(cfg.dataset, "timezone", None)
    if _tz and _tz.upper() == "UTC":
        import pytz as _pytz
        _utc = _pytz.utc
        def _to_ns(date_str):
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            dt = _utc.localize(dt)
            return int(dt.timestamp() * 1_000_000_000)
    else:
        _to_ns = datetime_to_ns_time_US

    log("Building graphs...")
    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"

        timestamps = [date_start, date_stop]
        test_mode_set_done = False

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = _to_ns(start)
            end_ns_timestamp = _to_ns(stop)

            attack_index = 0
            mimicry_events = []
            for attack_tuple in cfg.dataset.attack_to_time_window:
                attack = attack_tuple[0]
                attack_start_time = _to_ns(attack_tuple[1])
                attack_end_time = _to_ns(attack_tuple[2])

                if mimicry_edge_num > 0 and (
                    attack_start_time >= start_ns_timestamp and attack_end_time <= end_ns_timestamp
                ):
                    log(
                        f"Insert mimicry events into attack {attack_index} when building graphs from {date_start} to {date_stop}"
                    )
                    mimicry_events.extend(attack_mimicry_events[attack_index])
                attack_index += 1

            sql = """
            select * from event_table
            where
                  timestamp_rec > %s and timestamp_rec < %s
                   ORDER BY timestamp_rec, event_uuid;
            """
            sql_params = (start_ns_timestamp, end_ns_timestamp)

            BATCH = 1024
            # Fetch large DB chunks to minimise round-trips; no per-row generator
            # overhead — events are processed directly from list slices.
            FETCH_SIZE = 500_000
            window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

            start_time = None
            temp_list = []
            # Filtered events from the previous DB chunk that did not fill a
            # complete BATCH; prepended to the next chunk's filtered events.
            leftover = []

            cur = connect.cursor(_next_cursor_name())
            try:
                cur.execute(sql, sql_params)
                db_done = False
                test_mode_stop = False
                while not db_done and not test_mode_stop:
                    db_chunk = cur.fetchmany(FETCH_SIZE)
                    db_done = len(db_chunk) < FETCH_SIZE

                    # Take ownership of leftover, then append newly filtered rows.
                    filtered = leftover
                    leftover = []
                    for row in db_chunk:
                        if row[2] in include_edge_type:
                            filtered.append(row)

                    # On the last DB chunk also append mimicry events, matching
                    # the original events_list ordering (real events then mimicry).
                    if db_done:
                        for event in mimicry_events:
                            if event[2] in include_edge_type:
                                filtered.append(event)

                    if not filtered:
                        continue

                    # Initialise start_time from the very first event seen.
                    if start_time is None:
                        start_time = filtered[0][-2]

                    # Slice filtered into BATCH-sized chunks.  Any incomplete tail
                    # is saved as leftover for the next DB chunk (unless db_done).
                    n = len(filtered)
                    pos = 0
                    while pos < n:
                        batch_end = pos + BATCH
                        is_last_batch = db_done and batch_end >= n

                        if batch_end > n and not is_last_batch:
                            # Incomplete batch with more DB data coming; defer.
                            leftover = filtered[pos:]
                            break

                        batch_edges = filtered[pos : min(batch_end, n)]
                        pos = min(batch_end, n)

                        temp_list.extend(batch_edges)

                        if (batch_edges[-1][-2] > start_time + window_size_in_ns) or is_last_batch:
                            time_interval = (
                                ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone)
                                + "~"
                                + ns_time_to_datetime_US(batch_edges[-1][-2], timezone=cfg.dataset.timezone)
                            )

                            # log(f"Start create edge fused time window graph for {time_interval}")

                            node_info = {}
                            edge_list = []
                            if cfg.construction.fuse_edge:
                                edge_info = defaultdict(list)
                                for (
                                    src_node,
                                    src_index_id,
                                    operation,
                                    dst_node,
                                    dst_index_id,
                                    event_uuid,
                                    timestamp_rec,
                                    _id,
                                ) in temp_list:
                                    # Skip edges referencing filtered-out nodes
                                    if src_index_id not in indexid2msg or dst_index_id not in indexid2msg:
                                        continue
                                    if src_index_id not in node_info:
                                        node_type, label = indexid2msg[src_index_id]
                                        node_info[src_index_id] = {
                                            "label": label,
                                            "node_type": node_type,
                                        }
                                    if dst_index_id not in node_info:
                                        node_type, label = indexid2msg[dst_index_id]
                                        node_info[dst_index_id] = {
                                            "label": label,
                                            "node_type": node_type,
                                        }

                                    edge_info[(src_index_id, dst_index_id)].append(
                                        (timestamp_rec, operation, event_uuid)
                                    )

                                for (src, dst), data in edge_info.items():
                                    sorted_data = sorted(data, key=lambda x: x[0])
                                    operation_list = [entry[1] for entry in sorted_data]

                                    indices = []
                                    current_type = None
                                    current_start_index = None

                                    for idx, item in enumerate(operation_list):
                                        if item == current_type:
                                            continue
                                        else:
                                            if current_type is not None and current_start_index is not None:
                                                indices.append(current_start_index)
                                            current_type = item
                                            current_start_index = idx

                                    if current_type is not None and current_start_index is not None:
                                        indices.append(current_start_index)

                                    for k in indices:
                                        edge_list.append(
                                            {
                                                "src": src,
                                                "dst": dst,
                                                "time": sorted_data[k][0],
                                                "label": sorted_data[k][1],
                                                "event_uuid": sorted_data[k][2],
                                            }
                                        )
                            else:
                                for (
                                    src_node,
                                    src_index_id,
                                    operation,
                                    dst_node,
                                    dst_index_id,
                                    event_uuid,
                                    timestamp_rec,
                                    _id,
                                ) in temp_list:
                                    # Skip edges referencing filtered-out nodes
                                    if src_index_id not in indexid2msg or dst_index_id not in indexid2msg:
                                        continue
                                    if src_index_id not in node_info:
                                        node_type, label = indexid2msg[src_index_id]
                                        node_info[src_index_id] = {
                                            "label": label,
                                            "node_type": node_type,
                                        }
                                    if dst_index_id not in node_info:
                                        node_type, label = indexid2msg[dst_index_id]
                                        node_info[dst_index_id] = {
                                            "label": label,
                                            "node_type": node_type,
                                        }

                                    edge_list.append(
                                        {
                                            "src": src_index_id,
                                            "dst": dst_index_id,
                                            "time": timestamp_rec,
                                            "label": operation,
                                            "event_uuid": event_uuid,
                                        }
                                    )

                            # log(f"Start creating graph for {time_interval}")
                            graph = nx.MultiDiGraph()

                            for node, info in node_info.items():
                                graph.add_node(node, node_type=info["node_type"], label=info["label"])

                            # Translate OpTC edge types to DARPA TC equivalents if enabled
                            _translate = (
                                cfg.dataset.name in OPTC_DATASETS
                                and _has_consistent_edge_types(cfg)
                            )

                            for i, edge in enumerate(edge_list):
                                label = edge["label"]
                                if _translate:
                                    src_type = node_info[edge["src"]]["node_type"]
                                    dst_type = node_info[edge["dst"]]["node_type"]
                                    label = translate_optc_to_tc(label, src_type, dst_type)
                                graph.add_edge(
                                    edge["src"],
                                    edge["dst"],
                                    event_uuid=edge["event_uuid"],
                                    time=edge["time"],
                                    label=label,
                                    y=0,
                                )

                                # For unit tests, we only want few edges
                                NUM_TEST_EDGES = 2000
                                if cfg._test_mode and i >= NUM_TEST_EDGES:
                                    break

                            date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
                            os.makedirs(date_dir, exist_ok=True)
                            graph_name = f"{date_dir}/{time_interval}"

                            # log(f"Saving graph for {time_interval}")
                            torch.save(graph, graph_name)

                            # log(f"[{time_interval}] Num of edges: {len(edge_list)}")
                            # log(f"[{time_interval}] Num of events: {len(temp_list)}")
                            # log(f"[{time_interval}] Num of nodes: {len(node_info.keys())}")
                            start_time = batch_edges[-1][-2]
                            temp_list.clear()

                            # For unit tests, we only edges from the first graph
                            if cfg._test_mode:
                                test_mode_set_done = True
                                test_mode_stop = True
                                break
            finally:
                cur.close()


def main(cfg):
    """Main construction pipeline: build graphs from database and save metadata.

    Execution flow:
    1. Extract node features from database (compute_indexid2msg)
    2. Build time-windowed graphs from events (gen_edge_fused_tw)
    3. Compute dataset split node memberships (compute_and_save_split2nodes)
    4. Save filtered node features (save_indexid2msg)

    Args:
        cfg: Configuration object with all construction parameters
    """
    log_start(__file__)

    indexid2msg = compute_indexid2msg(cfg=cfg)

    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg)

    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)
