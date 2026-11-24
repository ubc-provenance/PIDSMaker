"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation.
"""

import gc
import multiprocessing
import os
import queue
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta

import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_dates_from_cfg, get_node_feats_from_cfg
from pidsmaker.config.schemas import benign_training
from pidsmaker.preprocessing.cleaning import Cleaner
from pidsmaker.utils.graph_store import save_edge_list
from pidsmaker.utils.dataset_utils import get_schema, iter_nodes
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


# Rows keep the shape of `select *` (src_node, src_index_id, operation, dst_node,
# dst_index_id, event_uuid, timestamp_rec, _id), but the columns that construction does
# not use are not transferred (NULL): fetching rows dominates the query time.
# The columns of the schema's edge attributes, then its label column, follow (see `events_sql`).
EVENT_TIME_IDX = 6
EVENT_EXTRA_IDX = 8


def events_sql(schema):
    """Query of the events of a time range, ordered by time (see the row shape above)."""
    extra = [attr.column for attr in schema.edge_attributes]
    if schema.label_column:
        extra.append(schema.label_column)
    extra_sql = "".join(f", {col}" for col in extra)
    return f"""
select NULL, src_index_id, operation, NULL, dst_index_id, event_uuid, timestamp_rec, NULL{extra_sql}
from {schema.event_table}
where
      timestamp_rec > %s and timestamp_rec < %s{schema.operations_sql()}
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
                cur.execute(events_sql(get_schema(cfg)), params)
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

    Reads all nodes of the dataset (the tables of its schema's node types) and builds the
    label of each node from its attributes, based on `construction.node_label_features`.

    Args:
        cfg: Configuration with database connection and feature settings
        cleaner: Optional `Cleaner` applied to node attributes and labels. Nodes it
            removes (noisy netflows) are absent from the mapping, so their events are skipped.

    Returns:
        dict: Mapping {index_id: [node_type, label_string]} where:
            - index_id: Database node identifier (str)
            - node_type: Name of the node type in the dataset's schema
            - label_string: Feature label (hashed or plaintext depending on config)
    """
    _, connect = init_database_connection(cfg)
    if cleaner is None:
        cleaner = Cleaner(cfg)

    schema = get_schema(cfg)
    use_hashed_label = cfg.construction.use_hashed_label
    null_label_tokens = cfg.construction.null_label_tokens
    node_label_features = get_node_feats_from_cfg(cfg)
    indexid2msg = {}

    def get_label_str_from_features(attrs, node_type):
        """Feature label of a node from its attributes and the configured features."""
        feats = node_label_features[node_type]
        if feats == ["auto"]:
            label_str = schema.node_type(node_type).auto_label_str(attrs, null_label_tokens)
        elif cleaner:
            # Null attributes were emptied by the cleaning: they are skipped
            label_str = " ".join([attrs[label_used] for label_used in feats if attrs[label_used]])
        else:
            label_str = " ".join([attrs[label_used] for label_used in feats])

        label_str = cleaner.normalize_label(label_str)
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        # Keep only printable ASCII — labels are paths, cmds, IPs, host names, ...
        label_str = re.sub(r"[^\x20-\x7e]+", "", label_str).strip()
        return label_str

    counts = Counter()
    for node_type, _uuid, _hash_id, index_id, attrs in iter_nodes(cfg, connect):
        if "remote_ip" in attrs and cleaner.is_noisy_netflow(attrs):
            continue
        attrs = cleaner.clean_attributes(node_type, attrs)
        label_str = get_label_str_from_features(attrs, node_type)
        indexid2msg[str(index_id)] = [node_type, label_str]
        counts[node_type] += 1

    for node_type, count in counts.items():
        log(f"Number of {node_type} nodes: {count}")
    connect.close()

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
    schema = get_schema(cfg)
    # DB rows carry raw edge type strings (e.g. "START", "MESSAGE" for OpTC), which are
    # filtered against the raw vocabulary when the schema translates edge types.
    include_edge_type = schema.event_filter_rel2id

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
    # The same date may appear in several splits (e.g. PROVENANCE_BENIGN): it is then built
    # several times as before, which parallel workers would do concurrently on the same files
    if num_workers > 1 and _can_fork() and len(set(dates)) == len(dates):
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
            cur.execute(events_sql(get_schema(cfg)), params)
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


class _EncodedValues(dict):
    """{database value: `attr.encode(value)`}, filled on first access (few distinct values)."""

    def __init__(self, attr):
        super().__init__()
        self.attr = attr

    def __missing__(self, value):
        encoded = self[value] = self.attr.encode(value)
        return encoded


def _attributes_encoder(edge_attributes):
    """Function of the extra columns of an event row returning the tuple of the encoded values
    of `edge_attributes` (the first columns), as `tuple(attr.encode(v) for ...)`."""
    encoders = [attr.encode for attr in edge_attributes]
    # Categorical attributes have few distinct values: their encodings are memoized
    getters = [
        _EncodedValues(attr).__getitem__ if attr.kind == "categorical" else attr.encode
        for attr in edge_attributes
    ]

    def encode_slow(extra):
        return tuple(encode(v) for encode, v in zip(encoders, extra))

    def encode_fast(extra):
        # One dict lookup per value instead of a call to `encode`
        return tuple([get(v) for get, v in zip(getters, extra)])

    def encode(extra):
        try:
            return encode_fast(extra)
        except TypeError:  # unhashable values
            return encode_slow(extra)

    return encode


def _save_window_graph(path, node_info, edge_list, edge_attr_names):
    """Saves the graph of a time window: nodes {node: [node_type, label]} and edges
    (src, dst, time, label, event_uuid, attribute values, y), with the edge attributes
    {"event_uuid", "time", "label", "y", **attributes} (attributes of the same name as the
    first ones replace them, as in a dict).

    The compact file is written from the edge list (`save_edge_list`), without building the
    networkx graph, which is slow for the millions of edges of the windows of network datasets.
    """
    infos = list(node_info.values())
    node_attrs = {"node_type": [i[0] for i in infos], "label": [i[1] for i in infos]}
    srcs, dsts, times, labels, uuids, attrs, ys = zip(*edge_list) if edge_list else ((),) * 7
    edge_attrs = {"event_uuid": uuids, "time": times, "label": labels, "y": ys}
    attr_cols = list(zip(*attrs)) if attrs else [()] * len(edge_attr_names)
    for name, col in zip(edge_attr_names, attr_cols):
        edge_attrs[name] = col
    save_edge_list(path, list(node_info), node_attrs, srcs, dsts, edge_attrs)


def _build_day(
    cfg, day_idx, date, day_params, day_chunks, fetch_size, indexid2msg, cleaner, use_cleaner,
    include_edge_type, attack_mimicry_events, mimicry_edge_num, _to_ns, file2nodes,
):
    """Builds and saves the time window graphs of a day (see `gen_edge_fused_tw`).

    `day_chunks` yields the `fetchmany(fetch_size)` chunks of the day's events query.
    """
    # The millions of event tuples of a window trigger costly cyclic GC passes that cannot free
    # anything (events hold no reference cycles)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        _build_day_events(
            cfg, day_idx, date, day_params, day_chunks, fetch_size, indexid2msg, cleaner,
            use_cleaner, include_edge_type, attack_mimicry_events, mimicry_edge_num, _to_ns,
            file2nodes,
        )
    finally:
        if gc_enabled:
            gc.enable()


def _build_day_events(
    cfg, day_idx, date, day_params, day_chunks, fetch_size, indexid2msg, cleaner, use_cleaner,
    include_edge_type, attack_mimicry_events, mimicry_edge_num, _to_ns, file2nodes,
):
    schema = get_schema(cfg)
    edge_attributes = schema.edge_attributes
    edge_attr_names = [attr.name for attr in edge_attributes]
    num_extras = len(edge_attributes) + (1 if schema.label_column else 0)
    has_extras = num_extras > 0
    edge_type_idx = (
        edge_attr_names.index(schema.edge_type_attribute) if schema.edge_type_attribute else None
    )

    encode_attrs = _attributes_encoder(edge_attributes)
    has_label = bool(schema.label_column)

    # With `benign_training`, the malicious events of train/val dates are dropped
    train_val_dates = set(cfg.dataset.train_dates) | set(cfg.dataset.val_dates)
    drop_malicious = (
        has_label
        and benign_training(cfg.construction)
        and date in train_val_dates
        and date not in set(cfg.dataset.test_dates)
    )
    label_idx = EVENT_EXTRA_IDX + len(edge_attributes)
    num_dropped_malicious = 0

    def read_extras(row):
        """(edge attribute values, label) of an event row. Missing columns (e.g. in events
        generated by mimicry) are null."""
        extra = row[EVENT_EXTRA_IDX:]
        if len(extra) < num_extras:
            extra = tuple(extra) + (None,) * (num_extras - len(extra))
        y = int(extra[-1] or 0) if has_label else 0
        return encode_attrs(extra), y

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
            chunk = [row for row in db_chunk if row[2] in include_edge_type]
            if drop_malicious:
                benign = [row for row in chunk if not row[label_idx]]
                num_dropped_malicious += len(chunk) - len(benign)
                chunk = benign
            filtered.extend(cleaner.filter_events(chunk) if use_cleaner else chunk)

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
                start_time = filtered[0][EVENT_TIME_IDX]

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

                if (batch_edges[-1][EVENT_TIME_IDX] > start_time + window_size_in_ns) or is_last_batch:
                    time_interval = (
                        ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone)
                        + "~"
                        + ns_time_to_datetime_US(batch_edges[-1][EVENT_TIME_IDX], timezone=cfg.dataset.timezone)
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
                    for row in temp_list:
                        src_index_id = row[1]
                        operation = row[2]
                        dst_index_id = row[4]
                        event_uuid = row[5]
                        timestamp_rec = row[EVENT_TIME_IDX]
                        if src_index_id not in node_info or dst_index_id not in node_info:
                            src_msg = indexid2msg.get(src_index_id)
                            dst_msg = indexid2msg.get(dst_index_id)
                            # Skip edges referencing filtered-out nodes
                            if src_msg is None or dst_msg is None:
                                continue
                            # node_info values are [node_type, label]
                            node_info.setdefault(src_index_id, src_msg)
                            node_info.setdefault(dst_index_id, dst_msg)

                        # Edge attributes and label (empty for datasets without them)
                        if has_extras:
                            attrs, y = read_extras(row)
                            if edge_type_idx is not None:
                                operation = attrs[edge_type_idx]
                        else:
                            attrs, y = (), 0

                        if fuse_edge:
                            edge_info[(src_index_id, dst_index_id)].append(
                                (timestamp_rec, operation, event_uuid, attrs, y)
                            )
                        else:
                            edge_list.append(
                                (src_index_id, dst_index_id, timestamp_rec, operation, event_uuid, attrs, y)
                            )

                    # Fusion: among the time-sorted events between two nodes, only the
                    # first event of each run of consecutive identical events (same operation,
                    # attributes and label) is kept
                    for (src, dst), data in edge_info.items():
                        sorted_data = sorted(data, key=lambda x: x[0])
                        current_key = None
                        for timestamp_rec, operation, event_uuid, attrs, y in sorted_data:
                            key = (operation, attrs, y)
                            if key != current_key:
                                current_key = key
                                edge_list.append(
                                    (src, dst, timestamp_rec, operation, event_uuid, attrs, y)
                                )

                    if cleaner:
                        cleaner.add_graph_nodes(node_info.keys())

                    # For unit tests, we only want few edges
                    NUM_TEST_EDGES = 2000
                    if cfg._test_mode:
                        edge_list = edge_list[: NUM_TEST_EDGES + 1]

                    # Translates edge types (e.g. OpTC to DARPA TC with consistent_edge_types)
                    if schema.edge_type_translation:
                        edge_list = [
                            (
                                src,
                                dst,
                                t,
                                schema.translate_edge_type(
                                    label, node_info[src][0], node_info[dst][0]
                                ),
                                uuid,
                                attrs,
                                y,
                            )
                            for src, dst, t, label, uuid, attrs, y in edge_list
                        ]
                    date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
                    os.makedirs(date_dir, exist_ok=True)
                    graph_name = f"{date_dir}/{time_interval}"

                    # log(f"Saving graph for {time_interval}")
                    _save_window_graph(graph_name, node_info, edge_list, edge_attr_names)
                    if file2nodes is not None:
                        file2nodes[os.path.normpath(graph_name)] = set(node_info)

                    # log(f"[{time_interval}] Num of edges: {len(edge_list)}")
                    # log(f"[{time_interval}] Num of events: {len(temp_list)}")
                    # log(f"[{time_interval}] Num of nodes: {len(node_info.keys())}")
                    start_time = batch_edges[-1][EVENT_TIME_IDX]
                    temp_list.clear()

                    # For unit tests, we only edges from the first graph
                    if cfg._test_mode:
                        test_mode_set_done = True
                        test_mode_stop = True
                        break

    if num_dropped_malicious:
        log(f"{date}: dropped {num_dropped_malicious} malicious events (benign training)")


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
