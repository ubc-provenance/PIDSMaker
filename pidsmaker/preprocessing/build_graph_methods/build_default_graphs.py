"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation.
"""

import multiprocessing
import os
import pickle
import queue
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta

import networkx as nx
import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.preprocessing.cleaning import Cleaner
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



# Rows keep the shape of `select *` (src_node, src_index_id, operation, dst_node,
# dst_index_id, event_uuid, timestamp_rec, _id), but the columns that construction does
# not use are not transferred (NULL): fetching rows dominates the query time.
EVENTS_SQL = """
select NULL, src_index_id, operation, NULL, dst_index_id, event_uuid, timestamp_rec, NULL
from event_table
where
      timestamp_rec > %s and timestamp_rec < %s
       ORDER BY timestamp_rec, event_uuid;
"""


class DayQueryPrefetcher:
    """Runs the daily event queries ahead of their processing.

    Each day is queried through its own connection and server-side cursor in a
    background thread, so that the database scans and sorts the next days while the
    current day is turned into graphs (daily queries dominate construction time).

    Memory stays bounded: each day buffers at most `queue_size` chunks of `fetch_size`
    rows, and at most `num_workers` days are fetched at the same time. Chunks are the
    exact `fetchmany` results (the last one being shorter than `fetch_size`).
    """

    def __init__(self, cfg, params_list, fetch_size, num_workers=4, queue_size=1):
        self.fetch_size = fetch_size
        self._stop = threading.Event()
        self._abandoned = [threading.Event() for _ in params_list]
        self._queues = [queue.Queue(maxsize=queue_size) for _ in params_list]
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_workers))
        # Days are submitted in order, so they are fetched in processing order
        for i, params in enumerate(params_list):
            self._pool.submit(self._produce, cfg, params, i)

    def _put(self, i, item):
        while not (self._stop.is_set() or self._abandoned[i].is_set()):
            try:
                self._queues[i].put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self, cfg, params, i):
        if self._stop.is_set() or self._abandoned[i].is_set():
            return
        connect = None
        try:
            _, connect = init_database_connection(cfg)
            with connect.cursor(_next_cursor_name()) as cur:
                cur.execute(EVENTS_SQL, params)
                while True:
                    chunk = cur.fetchmany(self.fetch_size)
                    if not self._put(i, chunk) or len(chunk) < self.fetch_size:
                        break
        except Exception as e:  # re-raised in the consumer thread
            self._put(i, e)
        finally:
            if connect is not None:
                connect.close()

    def chunks(self, i):
        """Yields the row chunks of the i-th day, in order."""
        while True:
            item = self._queues[i].get()
            if isinstance(item, Exception):
                raise item
            yield item
            if len(item) < self.fetch_size:
                return

    def abandon(self, i):
        """Stops fetching the i-th day (e.g. when its remaining rows are not needed)."""
        self._abandoned[i].set()
        self._drain(i)

    def _drain(self, i):
        try:
            while True:
                self._queues[i].get_nowait()
        except queue.Empty:
            pass

    def close(self):
        self._stop.set()
        for i in range(len(self._queues)):
            self._drain(i)
        self._pool.shutdown(wait=True)


def compute_indexid2msg(cfg, cleaner=None):
    """Compute mapping from node index IDs to node types and feature labels.

    Queries PostgreSQL database for all nodes (netflow, subject/process, file) and
    extracts their attributes to create feature labels based on configuration.

    Args:
        cfg: Configuration with database connection and feature settings
        cleaner: Optional `Cleaner` applied to node attributes and labels. Nodes it
            removes (noisy netflows) are absent from the mapping, so their events are skipped.

    Returns:
        dict: Mapping {index_id: [node_type, label_string]} where:
            - index_id: Database node identifier
            - node_type: One of 'netflow', 'subject', 'file'
            - label_string: Feature label (hashed or plaintext depending on config)
    """
    _, connect = init_database_connection(cfg)
    if cleaner is None:
        cleaner = Cleaner(cfg)

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
        elif cleaner:
            # Null attributes were emptied by the cleaning: they are skipped
            label_str = " ".join([attrs[label_used] for label_used in feats if attrs[label_used]])
        else:
            label_str = " ".join([attrs[label_used] for label_used in feats])

        label_str = cleaner.normalize_label(label_str)
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
        if cleaner.is_noisy_netflow(attrs):
            continue
        attrs = cleaner.clean_attributes(node_type, attrs)
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
        attrs = cleaner.clean_attributes(node_type, attrs)
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
        attrs = cleaner.clean_attributes(node_type, attrs)
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
    empty_splits = [split for split in ["train", "val", "test"] if not split2nodes.get(split)]
    if empty_splits:
        raise ValueError(
            f"No graph was built for the {empty_splits} split(s) of {cfg.dataset.name}. "
            "Check that the database contains events for the dates of these splits."
        )
    all_nodes = set().union(*(split2nodes[split] for split in ["train", "val", "test"]))
    indexid2msg = {k: v for k, v in indexid2msg.items() if k in all_nodes}

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg, file2nodes=None):
    """Compute and save mapping of dataset splits to their node sets.

    Loads all graphs from train/val/test splits and collects unique node IDs
    appearing in each split. Used to filter node features and track split membership.

    Args:
        cfg: Configuration with graph directory and split file paths
        file2nodes: Optional {normalized graph path: node set} recorded while building
            the graphs. Graphs found there are not reloaded from disk.

    Returns:
        dict: Mapping of split names to node sets:
            {'train': {node_ids}, 'val': {node_ids}, 'test': {node_ids}}
    """
    split_to_files = get_split_to_files(cfg, cfg.construction._graphs_dir)
    split2nodes = defaultdict(set)
    file2nodes = file2nodes or {}

    for split, files in split_to_files.items():
        for path in log_tqdm(files, desc=f"Check nodes in {split} set"):
            nodes = file2nodes.get(os.path.normpath(path))
            if nodes is None:
                nodes = torch.load(path).nodes()
            split2nodes[split].update(nodes)
    split2nodes = dict(split2nodes)

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


def gen_edge_fused_tw(indexid2msg, cfg, cleaner=None, file2nodes=None):
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
        cleaner: Optional `Cleaner` removing events (self-loops, duplicates, ...)
        file2nodes: Optional dict filled with {normalized graph path: node set}, so that
            the nodes of each graph can be retrieved without reloading it from disk.
    """
    if cleaner is None:
        cleaner = Cleaner(cfg)
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

    use_cleaner = bool(cleaner)
    # Fetch large DB chunks to minimise round-trips; no per-row generator
    # overhead — events are processed directly from list slices.
    FETCH_SIZE = 500_000
    day_params = []
    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"
        day_params.append((_to_ns(date_start), _to_ns(date_stop)))
    common = dict(
        cfg=cfg,
        day_params=day_params,
        fetch_size=FETCH_SIZE,
        indexid2msg=indexid2msg,
        cleaner=cleaner,
        use_cleaner=use_cleaner,
        include_edge_type=include_edge_type,
        attack_mimicry_events=attack_mimicry_events,
        mimicry_edge_num=mimicry_edge_num,
        _to_ns=_to_ns,
    )
    if file2nodes is None:
        file2nodes = {}

    num_workers = min(get_num_workers(cfg), len(dates))
    if num_workers > 1 and _can_fork():
        log(f"Building graphs ({num_workers} processes)...")
        _build_days_in_parallel(dates, common, num_workers, file2nodes)
    else:
        log("Building graphs...")
        # Days are processed sequentially, while the next ones are queried in the background
        prefetcher = DayQueryPrefetcher(cfg, day_params, FETCH_SIZE)
        try:
            for day_idx, date in enumerate(dates):
                try:
                    _build_day(
                        day_idx=day_idx,
                        date=date,
                        day_chunks=prefetcher.chunks(day_idx),
                        file2nodes=file2nodes,
                        **common,
                    )
                finally:
                    prefetcher.abandon(day_idx)
        finally:
            prefetcher.close()


def get_num_workers(cfg):
    """Number of processes for CPU-bound preprocessing (`--num_workers`, default: up to 8)."""
    num_workers = getattr(cfg, "_num_workers", None)
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    return max(1, int(num_workers))


def _can_fork():
    """Workers are forked to share the (large) read-only state without pickling it.

    Forking after CUDA initialization is unsafe, in which case days are built sequentially.
    """
    if sys.platform == "win32" or "fork" not in multiprocessing.get_all_start_methods():
        return False
    try:
        import torch.cuda

        return not torch.cuda.is_initialized()
    except Exception:
        return False


# State shared with the forked workers of `_build_days_in_parallel` (inherited, not pickled)
_DAY_WORKER_STATE = None


def _iter_day_chunks(cfg, params, fetch_size):
    """Yields the `fetchmany` chunks of the events query of a day, on a new connection."""
    _, connect = init_database_connection(cfg)
    try:
        with connect.cursor(_next_cursor_name()) as cur:
            cur.execute(EVENTS_SQL, params)
            while True:
                chunk = cur.fetchmany(fetch_size)
                yield chunk
                if len(chunk) < fetch_size:
                    return
    finally:
        connect.close()


def _day_worker(day_idx, date):
    common = _DAY_WORKER_STATE
    # Only the per-day cleaning state is collected, then merged by the parent. Days are
    # independent: their events never share a timestamp (bounds are exclusive).
    cleaner = common["cleaner"]
    cleaner.stats = Counter()
    cleaner._seen_nodes = set()
    cleaner._graph_nodes = set()
    cleaner._last_timestamp = None
    cleaner._events_at_timestamp = set()

    file2nodes = {}
    day_chunks = _iter_day_chunks(common["cfg"], common["day_params"][day_idx], common["fetch_size"])
    try:
        _build_day(day_idx=day_idx, date=date, day_chunks=day_chunks, file2nodes=file2nodes, **common)
    finally:
        day_chunks.close()
    return file2nodes, cleaner.stats, cleaner._seen_nodes, cleaner._graph_nodes


def _build_days_in_parallel(dates, common, num_workers, file2nodes):
    """Builds the graphs of each day in a separate process (results are identical)."""
    global _DAY_WORKER_STATE
    _DAY_WORKER_STATE = common
    cleaner = common["cleaner"]
    try:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_day_worker, i, date) for i, date in enumerate(dates)]
            for future in futures:  # merged in day order
                day_file2nodes, stats, seen_nodes, graph_nodes = future.result()
                file2nodes.update(day_file2nodes)
                cleaner.stats.update(stats)
                cleaner._seen_nodes |= seen_nodes
                cleaner._graph_nodes |= graph_nodes
    finally:
        _DAY_WORKER_STATE = None


def _build_day(
    cfg, day_idx, date, day_params, day_chunks, fetch_size, indexid2msg, cleaner, use_cleaner,
    include_edge_type, attack_mimicry_events, mimicry_edge_num, _to_ns, file2nodes,
):
    """Builds and saves the time window graphs of a day (see `gen_edge_fused_tw`).

    `day_chunks` yields the `fetchmany(fetch_size)` chunks of the day's events query.
    """
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

        assert (start_ns_timestamp, end_ns_timestamp) == day_params[day_idx]
        FETCH_SIZE = fetch_size

        BATCH = 1024
        window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

        start_time = None
        temp_list = []
        # Filtered events from the previous DB chunk that did not fill a
        # complete BATCH; prepended to the next chunk's filtered events.
        leftover = []

        db_done = False
        test_mode_stop = False
        while not db_done and not test_mode_stop:
            db_chunk = next(day_chunks)
            db_done = len(db_chunk) < FETCH_SIZE

            # Take ownership of leftover, then append newly filtered rows.
            filtered = leftover
            leftover = []
            if use_cleaner:
                keep_event = cleaner.keep_event
                filtered.extend(
                    row
                    for row in db_chunk
                    if row[2] in include_edge_type
                    and keep_event(row[1], row[2], row[4], row[6])
                )
            else:
                filtered.extend(row for row in db_chunk if row[2] in include_edge_type)

            # On the last DB chunk also append mimicry events, matching
            # the original events_list ordering (real events then mimicry).
            if db_done:
                for event in mimicry_events:
                    if event[2] in include_edge_type and (
                        not cleaner
                        or cleaner.keep_event(event[1], event[2], event[4], event[6])
                    ):
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
                    # Edges as (src, dst, time, operation, event_uuid) tuples
                    edge_list = []
                    # (src_index_id, operation, dst_index_id) of each event
                    temp_list = cleaner.dedup_window(
                        temp_list, key_fn=lambda e: (e[1], e[2], e[4])
                    )
                    fuse_edge = cfg.construction.fuse_edge
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
                        if src_index_id not in node_info or dst_index_id not in node_info:
                            src_msg = indexid2msg.get(src_index_id)
                            dst_msg = indexid2msg.get(dst_index_id)
                            # Skip edges referencing filtered-out nodes
                            if src_msg is None or dst_msg is None:
                                continue
                            # node_info values are [node_type, label]
                            node_info.setdefault(src_index_id, src_msg)
                            node_info.setdefault(dst_index_id, dst_msg)

                        if fuse_edge:
                            edge_info[(src_index_id, dst_index_id)].append(
                                (timestamp_rec, operation, event_uuid)
                            )
                        else:
                            edge_list.append(
                                (src_index_id, dst_index_id, timestamp_rec, operation, event_uuid)
                            )

                    # Fusion: among the time-sorted events between two nodes, only the
                    # first event of each run of consecutive identical operations is kept
                    for (src, dst), data in edge_info.items():
                        sorted_data = sorted(data, key=lambda x: x[0])
                        current_type = None
                        for timestamp_rec, operation, event_uuid in sorted_data:
                            if operation != current_type:
                                current_type = operation
                                edge_list.append(
                                    (src, dst, timestamp_rec, operation, event_uuid)
                                )

                    # log(f"Start creating graph for {time_interval}")
                    graph = nx.MultiDiGraph()
                    # node_info values are [node_type, label]
                    graph.add_nodes_from(
                        (node, {"node_type": info[0], "label": info[1]})
                        for node, info in node_info.items()
                    )
                    if cleaner:
                        cleaner.add_graph_nodes(node_info.keys())

                    # For unit tests, we only want few edges
                    NUM_TEST_EDGES = 2000
                    if cfg._test_mode:
                        edge_list = edge_list[: NUM_TEST_EDGES + 1]

                    # Translate OpTC edge types to DARPA TC equivalents if enabled
                    _translate = (
                        cfg.dataset.name in OPTC_DATASETS
                        and _has_consistent_edge_types(cfg)
                    )
                    if _translate:
                        edge_list = [
                            (
                                src,
                                dst,
                                t,
                                translate_optc_to_tc(
                                    label, node_info[src][0], node_info[dst][0]
                                ),
                                uuid,
                            )
                            for src, dst, t, label, uuid in edge_list
                        ]
                    graph.add_edges_from(
                        (src, dst, {"event_uuid": uuid, "time": t, "label": label, "y": 0})
                        for src, dst, t, label, uuid in edge_list
                    )

                    date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    # log(f"Saving graph for {time_interval}")
                    torch.save(graph, graph_name, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    if file2nodes is not None:
                        file2nodes[os.path.normpath(graph_name)] = set(node_info)

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


def main(cfg):
    """Main construction pipeline: build graphs from database and save metadata.

    Execution flow:
    1. Extract node features from database (compute_indexid2msg)
    2. Build time-windowed graphs from events (gen_edge_fused_tw)
    3. Compute dataset split node memberships (compute_and_save_split2nodes)
    4. Save filtered node features (save_indexid2msg)
    5. With `construction.cleaning`, save the cleaning report

    Args:
        cfg: Configuration object with all construction parameters
    """
    log_start(__file__)

    cleaner = Cleaner(cfg)
    if cleaner:
        log(f"Cleaning methods: {', '.join(sorted(cleaner.methods))}")

    indexid2msg = compute_indexid2msg(cfg=cfg, cleaner=cleaner)

    file2nodes = {}
    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg, cleaner=cleaner, file2nodes=file2nodes)

    split2nodes = compute_and_save_split2nodes(cfg, file2nodes=file2nodes)
    del file2nodes
    save_indexid2msg(indexid2msg, split2nodes, cfg)

    if cleaner:
        cleaner.save_report(cfg)
