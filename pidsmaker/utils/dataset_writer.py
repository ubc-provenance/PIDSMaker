"""Writing of Parquet datasets (format and order of rows: see `pidsmaker/utils/database.py`).

Shared by the scripts that build datasets from their raw files (`dataset_preprocessing/`), which
give rows from Python (`DatasetBuilder`), and by the export of Postgres databases
(`dataset_preprocessing/export_parquet.py`), so that both write the same files:

- a node table is a single file, whose rows are in the physical order of the Postgres table
  (the order of `SELECT * FROM table`). It is not exactly the order in which the rows were
  inserted, as Postgres fills the free space of earlier pages with short rows: the builder
  finds the position of each row as Postgres does (see `HeapLayout`);
- the event table is sorted by `timestamp_rec, event_uuid`, one file per day (UTC). DuckDB
  compares strings byte by byte, Postgres with the collation of the database (en_US.utf8, which
  ignores dashes and case at first). Both orders are the same when the event UUIDs pass
  `check_tie_order`, whose result is recorded in the manifest (`tie_order`). Events of a same
  timestamp and UUID (duplicate records), whose order Postgres does not define, are in the order
  of their `_id` in the datasets of the builder.

Values are those that Postgres stored with the scripts that loaded the datasets into Postgres
(before Parquet): the columns of a table keep the names, order and types of the Postgres table
(see `TableSpec`), and the serial column `_id` of event tables numbers the events from 1 in the
order in which they are given.
"""

import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass

EVENT_ORDER = "timestamp_rec, event_uuid"
DAY_NS = 86_400 * 10**9
# Event UUIDs whose byte order is the order of the collation of Postgres (see `check_tie_order`)
UUID_PATTERNS = {
    "uuid_upper": "[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}",
    "uuid_lower": "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    "alnum_upper": "[0-9A-Z]*",
    "alnum_lower": "[0-9a-z]*",
}
# Collation of the Postgres databases of the datasets, whose order the event files follow
COLLATION = "en_US.utf8"
ROW_GROUP_SIZE = 1_000_000

# Postgres types of the columns of the datasets (as named by information_schema), and their
# DuckDB types
VARCHAR, BIGINT, INTEGER, SMALLINT, DOUBLE = (
    "character varying",
    "bigint",
    "integer",
    "smallint",
    "double precision",
)
DUCKDB_TYPES = {
    VARCHAR: "VARCHAR",
    BIGINT: "BIGINT",
    INTEGER: "INTEGER",
    SMALLINT: "SMALLINT",
    DOUBLE: "DOUBLE",
}
# Largest value of a serial column (SERIAL, BIGSERIAL), whose sequence fails beyond
SERIAL_MAX = {INTEGER: 2**31 - 1, BIGINT: 2**63 - 1}


def sql_str(s):
    return "'" + s.replace("'", "''") + "'"


def sql_list(paths):
    return "[" + ", ".join(sql_str(p) for p in paths) + "]"


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def connect_duckdb(temp_dir, threads=None, memory_gb=None):
    """DuckDB connection that writes Parquet files: rows of queries without ORDER BY are kept in
    the order of their source, and large sorts spill to `temp_dir`."""
    import duckdb

    config = {"temp_directory": temp_dir, "preserve_insertion_order": True}
    if threads:
        config["threads"] = threads
    if memory_gb:
        config["memory_limit"] = f"{memory_gb}GB"
    return duckdb.connect(config=config)


def copy_to_parquet(db, query, path, row_group_size=ROW_GROUP_SIZE):
    """Writes the rows of `query` to the Parquet file `path`, in the order of the query."""
    db.execute(
        f"COPY ({query}) TO {sql_str(path)} "
        f"(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {row_group_size});"
    )


def count_rows(db, paths):
    return db.execute(f"SELECT count(*) FROM read_parquet({sql_list(paths)});").fetchone()[0]


def check_tie_order(db, events, log=print):
    """Whether the byte order of the event UUIDs of `events` (a DuckDB relation) is the order
    of the collation of Postgres, for the events of a same timestamp.

    Both orders are the same when all the event UUIDs are canonical UUIDs of a single case, or
    alphanumeric strings of a single case, or more generally when no letter has both cases and
    the events of each timestamp have IDs of the same shape (same length and separators at the
    same places, e.g. `agraph_12_0000001` in LANL). Returns the name of the check that passed,
    or `unverified`."""
    checks = ", ".join(
        f"count(*) FILTER (WHERE NOT regexp_full_match(event_uuid, '{p}'))"
        for p in UUID_PATTERNS.values()
    )
    mismatches = db.execute(f"SELECT {checks} FROM {events};").fetchone()
    patterns = [name for name, n in zip(UUID_PATTERNS, mismatches) if n == 0]
    if patterns:
        return patterns[0]
    mixed_case = db.execute(
        f"SELECT bool_or(regexp_matches(event_uuid, '[a-z]')) "
        f"AND bool_or(regexp_matches(event_uuid, '[A-Z]')) FROM {events};"
    ).fetchone()[0]
    mixed_shapes = db.execute(
        f"SELECT count(*) FROM (SELECT timestamp_rec FROM {events} GROUP BY timestamp_rec "
        "HAVING count(DISTINCT regexp_replace(event_uuid, '[0-9A-Za-z]', 'x', 'g')) > 1);"
    ).fetchone()[0]
    if not mixed_case and mixed_shapes == 0:
        return "same_shape_in_ties"
    log(
        f"WARNING: event UUIDs match none of {list(UUID_PATTERNS)}, letters of both cases: "
        f"{bool(mixed_case)}, timestamps with UUIDs of several shapes: {mixed_shapes}. The order "
        "of the events of a same timestamp may differ from Postgres"
    )
    return "unverified"


def write_events_by_day(
    db, source, columns, staging, table_dir, row_group_size=ROW_GROUP_SIZE, log=print, order=None
):
    """Writes the events of `source` (a DuckDB relation with the `columns` of the event table)
    sorted by `order` (default: `EVENT_ORDER`), one file per day (UTC) in `table_dir`.

    Events are copied unsorted, partitioned by day, in `staging` (removed at the end), then
    each day is sorted. Returns the description of the table in the manifest."""
    db.execute(
        f"COPY (SELECT *, timestamp_rec // {DAY_NS} AS _day FROM {source}) "
        f"TO {sql_str(staging)} (FORMAT parquet, COMPRESSION zstd, PARTITION_BY (_day));"
    )
    log("copied, sorting each day")
    stage = f"read_parquet({sql_str(staging + '/*/*.parquet')}, hive_partitioning = true)"

    tie_order = check_tie_order(db, stage, log)

    # Partition folders are `_day=<day>`, NULL timestamps (if any) in a non-numeric one, sorted
    # last as by ORDER BY in Postgres
    def day_key(folder):
        value = folder.split("=", 1)[1]
        return (0, int(value)) if value.lstrip("-").isdigit() else (1, 0)

    names = ", ".join(f'"{n}"' for n in columns)
    files = []
    for folder in sorted(os.listdir(staging), key=day_key):
        is_day, day = day_key(folder)
        name = (
            f"part-{datetime.datetime.utcfromtimestamp(day * 86400):%Y-%m-%d}.parquet"
            if is_day == 0
            else "part-null.parquet"
        )
        day_dir = os.path.join(staging, folder)
        copy_to_parquet(
            db,
            f"SELECT {names} FROM read_parquet({sql_str(day_dir + '/*.parquet')}) "
            f"ORDER BY {order or EVENT_ORDER}",
            os.path.join(table_dir, name),
            row_group_size,
        )
        files.append(name)
        shutil.rmtree(day_dir)
    shutil.rmtree(staging)
    return {"files": files, "order": f"ORDER BY {EVENT_ORDER}", "tie_order": tie_order}


def write_manifest(dataset_dir, database, source, tables, collation=COLLATION, exported_at=None):
    """Writes the manifest of a dataset. `tables` is {table: description} (files, order, number
    of rows and columns of each table)."""
    manifest = {
        "format_version": 1,
        "database": database,
        "source": source,
        "collation": collation,
        "exported_at": exported_at or utc_now(),
        "tables": tables,
    }
    with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------------------
# Datasets built from Python
# --------------------------------------------------------------------------------------


@dataclass
class TableSpec:
    """A table of a dataset.

    Attributes:
        name: name of the table.
        columns: [(column, Postgres type)] in the order of the table.
        not_null: columns that cannot be null (NOT NULL or primary key in Postgres).
        serial: column numbering the rows from 1 in the order in which they are given, as a
            SERIAL column of Postgres (filled by the writer, not given in rows), or None.
    """

    name: str
    columns: list
    not_null: tuple = ()
    serial: str = None

    @property
    def names(self):
        return [name for name, _ in self.columns]

    @property
    def row_columns(self):
        """[(column, type)] of the values of the rows given to the writer (all but `serial`)."""
        return [(name, kind) for name, kind in self.columns if name != self.serial]

    @property
    def is_event_table(self):
        return "timestamp_rec" in self.names and "event_uuid" in self.names


def node_table_spec(table, attribute_columns):
    """Node table: `node_uuid`, `hash_id`, the attribute columns (strings) and `index_id`."""
    columns = [("node_uuid", VARCHAR), ("hash_id", VARCHAR)]
    columns += [(column, VARCHAR) for column in attribute_columns] + [("index_id", BIGINT)]
    return TableSpec(table, columns, not_null=("node_uuid", "hash_id"))


def event_table_spec(table, extra_columns=(), serial_type=BIGINT):
    """Event table: the standard event columns, then `extra_columns` [(column, type)]."""
    columns = [
        ("src_node", VARCHAR),
        ("src_index_id", VARCHAR),
        ("operation", VARCHAR),
        ("dst_node", VARCHAR),
        ("dst_index_id", VARCHAR),
        ("event_uuid", VARCHAR),
        ("timestamp_rec", BIGINT),
        ("_id", serial_type),
    ] + list(extra_columns)
    return TableSpec(table, columns, not_null=("event_uuid",), serial="_id")


def schema_tables(schema, serial_type=BIGINT):
    """[TableSpec] of the tables of a dataset schema (see `pidsmaker/config/schemas.py`): a node
    table per node type, with the columns of its attributes, and the event table, with the
    columns of the edge attributes (strings for categorical ones, floats for numeric ones) and
    the label column. The `_id` of events is an `integer` (SERIAL) in the databases of the
    provenance datasets, a `bigint` (BIGSERIAL) in those of the network datasets."""
    tables = [node_table_spec(nt.table, [col for _, col in nt.columns]) for nt in schema.node_types]
    extra = [
        (a.column, VARCHAR if a.kind == "categorical" else DOUBLE) for a in schema.edge_attributes
    ]
    if schema.label_column:
        extra.append((schema.label_column, SMALLINT))
    tables.append(event_table_spec(schema.event_table, extra, serial_type))
    return tables


# --------------------------------------------------------------------------------------
# Physical order of the node tables
# --------------------------------------------------------------------------------------

# Constants of Postgres (8 kB pages, 8-byte MAXALIGN), see src/include/storage/bufpage.h,
# access/htup_details.h, storage/fsm_internals.h and access/heaptoast.h
BLCKSZ = 8192
PAGE_HEADER = 24
ITEM_ID = 4
TUPLE_HEADER = 23  # offsetof(HeapTupleHeaderData, t_bits)
MAX_HEAP_TUPLES_PER_PAGE = 291
TOAST_TUPLE_THRESHOLD = 2032
FSM_CAT_STEP = BLCKSZ // 256
MAX_FSM_REQUEST_SIZE = BLCKSZ - 32  # MaxHeapTupleSize
FSM_NODES = BLCKSZ - PAGE_HEADER - 4  # NodesPerPage
FSM_NON_LEAF_NODES = BLCKSZ // 2 - 1  # NonLeafNodesPerPage
FSM_SLOTS = FSM_NODES - FSM_NON_LEAF_NODES  # SlotsPerFSMPage: heap pages per FSM leaf page


def _maxalign(n):
    return (n + 7) & ~7


def heap_tuple_sizes(arrays, kinds):
    """Length (t_len) of the heap tuples of the rows of a Postgres table, from the Arrow
    arrays of its columns (`heap_form_tuple`): a header with the null bitmap, then the
    values, aligned by type. Strings of at most 126 bytes have a 1-byte header and no
    alignment, longer ones a 4-byte header and an alignment of 4 bytes. Values are not
    compressed (tuples longer than TOAST_TUPLE_THRESHOLD are compressed by Postgres)."""
    import numpy as np
    import pyarrow.compute as pc

    n = len(arrays[0])
    offset = np.zeros(n, dtype=np.int64)
    has_null = np.zeros(n, dtype=bool)
    for array, kind in zip(arrays, kinds):
        null = array.is_null().to_numpy(zero_copy_only=False)
        has_null |= null
        if kind == VARCHAR:
            length = pc.fill_null(pc.binary_length(array), 0).to_numpy().astype(np.int64)
            short = length <= 126
            value = np.where(short, offset + length + 1, ((offset + 3) & ~3) + length + 4)
        else:
            size = {BIGINT: 8, DOUBLE: 8, INTEGER: 4, SMALLINT: 2}[kind]
            value = ((offset + size - 1) & ~(size - 1)) + size
        offset = np.where(null, offset, value)
    bitmap = (len(kinds) + 7) // 8
    header = np.where(has_null, _maxalign(TUPLE_HEADER + bitmap), _maxalign(TUPLE_HEADER))
    return header + offset


class _FSMPage:
    """A page of the free space map (FSM): a binary tree whose leaves are the free space
    categories of the pages below it (heap pages, or FSM pages of the level below), and whose
    nodes are the maximum of their children (storage/freespace/fsmpage.c)."""

    def __init__(self):
        self.nodes = bytearray(FSM_NODES)
        self.next_slot = 0

    def set_avail(self, slot, value):
        nodes = self.nodes
        node = FSM_NON_LEAF_NODES + slot
        if nodes[node] == value and value <= nodes[0]:
            return
        nodes[node] = value
        while node > 0:
            node = (node - 1) // 2
            left = 2 * node + 1
            new = nodes[left]
            if left + 1 < FSM_NODES:
                new = max(new, nodes[left + 1])
            if nodes[node] == new:
                break
            nodes[node] = new

    def search_avail(self, min_value, advance_next):
        """Slot with at least `min_value`, the first one to the right of `next_slot` (wrapping
        around), or -1."""
        nodes = self.nodes
        if nodes[0] < min_value:
            return -1
        target = self.next_slot
        if target < 0 or target >= FSM_SLOTS:
            target = 0
        node = target + FSM_NON_LEAF_NODES
        while node > 0:
            if nodes[node] >= min_value:
                break
            # Move right, wrapping around to the leftmost node of the next level, and climb up
            node += 1
            if ((node + 1) & node) == 0:
                node = (node - 1) // 2
            node = (node - 1) // 2
        while node < FSM_NON_LEAF_NODES:
            child = 2 * node + 1
            if child < FSM_NODES and nodes[child] >= min_value:
                node = child
            elif child + 1 < FSM_NODES and nodes[child + 1] >= min_value:
                node = child + 1
            else:
                raise RuntimeError("Inconsistent free space map")
        slot = node - FSM_NON_LEAF_NODES
        self.next_slot = slot + (1 if advance_next else 0)
        return slot


def _avail_to_cat(avail):
    return 255 if avail >= MAX_FSM_REQUEST_SIZE else min(avail // FSM_CAT_STEP, 254)


def _needed_to_cat(needed):
    return 1 if needed == 0 else min((needed + FSM_CAT_STEP - 1) // FSM_CAT_STEP, 255)


class HeapLayout:
    """Positions (page, line) of the rows inserted into a new Postgres table, in a single
    session, which give its physical order: the order of `SELECT * FROM table`.

    Rows are not stored in the order of their insertion. When a row does not fit in the
    current page, Postgres records the free space of the page in the free space map (FSM),
    and looks in the FSM for a page with enough space before adding pages to the table
    (`RelationGetBufferForTuple` of access/heap/hio.c): a short row inserted after long ones
    can thus fill the end of an earlier page. This emulates Postgres 17 on 8 kB pages, for
    rows inserted by INSERT (`insert`, one row at a time), or by COPY (`multi_insert` of each
    batch of rows, written to pages added several at a time, see `CopyLayout`,
    `heap_multi_insert` of access/heap/heapam.c and `CopyFrom` of commands/copyfrom.c). The
    positions are those of Postgres for tables of hundreds of thousands of rows loaded both
    ways, and the physical order of the node tables of the DARPA TC and OpTC databases
    (restored from dumps by COPY) is the order given by both emulations in a row. Rows longer
    than TOAST_TUPLE_THRESHOLD, which Postgres compresses, are not supported."""

    def __init__(self, bulk=False):
        self.bulk = bulk  # COPY, with a BulkInsertState
        self.upper = []  # pd_upper of each page
        self.lines = []  # number of line pointers of each page
        self.fsm = {}  # FSM pages, by (level, logical page number)
        self.fsm_size = 0  # number of pages of the FSM
        self.target = None  # page of the last insertion
        self.next_free = None  # pages added in advance by COPY: next_free..last_free
        self.last_free = None
        self.extended_by = 0

    # Free space map (storage/freespace/freespace.c), whose tree has 3 levels of FSM pages

    @staticmethod
    def _fsm_block(level, page):
        """Block of an FSM page in the FSM (fsm_logical_to_physical)."""
        leaf = page * FSM_SLOTS**level
        pages = 0
        for _ in range(3):
            pages += leaf + 1
            leaf //= FSM_SLOTS
        return pages - level - 1

    def _fsm_page(self, level, page, extend):
        if self._fsm_block(level, page) >= self.fsm_size:
            if not extend:
                return None
            self.fsm_size = self._fsm_block(level, page) + 1
        fsm = self.fsm.get((level, page))
        if fsm is None:
            fsm = self.fsm[(level, page)] = _FSMPage()
        return fsm

    def _set_and_search(self, level, page, slot, value, min_value):
        fsm = self._fsm_page(level, page, extend=True)
        fsm.set_avail(slot, value)
        return fsm.search_avail(min_value, level == 0) if min_value else -1

    def _search(self, min_cat):
        """fsm_search: a heap page with a category of at least `min_cat`, or None."""
        level, page, restarts = 2, 0, 0
        while True:
            fsm = self._fsm_page(level, page, extend=False)
            slot = -1 if fsm is None else fsm.search_avail(min_cat, level == 0)
            if slot != -1:
                if level == 0:
                    return page * FSM_SLOTS + slot
                level, page = level - 1, page * FSM_SLOTS + slot
            elif level == 2:
                return None
            else:
                # The upper level was out of date: fix it and restart from the root
                max_avail = 0 if fsm is None else fsm.nodes[0]
                self._set_and_search(level + 1, page // FSM_SLOTS, page % FSM_SLOTS, max_avail, 0)
                restarts += 1
                if restarts > 10000:
                    return None
                level, page = 2, 0

    def _vacuum(self, level, page, start, end):
        """fsm_vacuum_page: updates the upper levels from the leaves for heap pages
        start..end-1. Returns (maximum category of the page, end of the FSM reached)."""
        fsm = self._fsm_page(level, page, extend=False)
        if fsm is None:
            return 0, True
        if level > 0:
            start_page, start_slot = divmod(start, FSM_SLOTS)
            end_page, end_slot = divmod(end - 1, FSM_SLOTS)
            for _ in range(level):
                start_page, start_slot = divmod(start_page, FSM_SLOTS)
                end_page, end_slot = divmod(end_page, FSM_SLOTS)
            if start_page != page:
                start_slot = FSM_SLOTS if start_page > page else 0
            if end_page != page:
                end_slot = FSM_SLOTS - 1 if end_page > page else -1
            eof = False
            for slot in range(start_slot, end_slot + 1):
                child = 0
                if not eof:
                    child, eof = self._vacuum(level - 1, page * FSM_SLOTS + slot, start, end)
                if fsm.nodes[FSM_NON_LEAF_NODES + slot] != child:
                    fsm.set_avail(slot, child)
        fsm.next_slot = 0
        return fsm.nodes[0], False

    def _record(self, heap_page, avail):
        """RecordPageWithFreeSpace."""
        page, slot = divmod(heap_page, FSM_SLOTS)
        self._set_and_search(0, page, slot, _avail_to_cat(avail), 0)

    def _record_and_search(self, heap_page, avail, needed):
        """RecordAndGetPageWithFreeSpace."""
        page, slot = divmod(heap_page, FSM_SLOTS)
        found = self._set_and_search(0, page, slot, _avail_to_cat(avail), _needed_to_cat(needed))
        if found != -1 and page * FSM_SLOTS + found < len(self.upper):
            return page * FSM_SLOTS + found
        return self._search(_needed_to_cat(needed))

    # Heap pages (access/heap/hio.c)

    def _free_space(self, page):
        """PageGetHeapFreeSpace."""
        if self.lines[page] >= MAX_HEAP_TUPLES_PER_PAGE:
            return 0
        return max(self.upper[page] - (PAGE_HEADER + ITEM_ID * self.lines[page]) - ITEM_ID, 0)

    def _add_pages(self, num_pages):
        """RelationAddBlocks: adds pages to the table, and returns the first one."""
        extend_by = num_pages
        if self.bulk:
            extend_by = max(extend_by, self.extended_by)
        extend_by = min(extend_by, 64)
        # Pages added for the rows to insert are not in the FSM
        not_in_fsm = 1 if num_pages > 1 and not self.bulk else num_pages
        first = len(self.upper)
        self.upper += [BLCKSZ] * extend_by
        self.lines += [0] * extend_by
        for i in range(max(not_in_fsm, 1), extend_by):
            self._record(first + i, BLCKSZ - PAGE_HEADER)
        if not_in_fsm < extend_by:
            self._vacuum(2, 0, first + not_in_fsm, first + extend_by)
        if self.bulk:
            if extend_by > 1:
                self.next_free, self.last_free = first + 1, first + extend_by - 1
            else:
                self.next_free = self.last_free = None
            self.extended_by += extend_by
        return first

    def _page_for(self, length, num_pages=1):
        """RelationGetBufferForTuple: page of a row whose heap tuple has `length` bytes."""
        needed = _maxalign(length)
        page = self.target
        if page is None:
            page = self._search(_needed_to_cat(needed))
        if page is None and self.upper:
            page = len(self.upper) - 1
        while page is not None:
            avail = self._free_space(page)
            if needed <= avail:
                self.target = page
                return page
            if self.bulk and self.next_free is not None:
                # Pages added in advance are used in order
                self._record(page, avail)
                page = self.next_free
                if self.next_free >= self.last_free:
                    self.next_free = self.last_free = None
                else:
                    self.next_free += 1
            else:
                page = self._record_and_search(page, avail, needed)
        self.target = self._add_pages(max(num_pages, 1))
        return self.target

    def _put(self, page, length):
        self.upper[page] -= _maxalign(length)
        self.lines[page] += 1
        return page, self.lines[page]

    def insert(self, length):
        """Position (page, line) of a row inserted by INSERT (heap_insert)."""
        return self._put(self._page_for(length), length)

    def multi_insert(self, lengths):
        """Positions of a batch of rows inserted by COPY (heap_multi_insert)."""
        positions, done, n = [], 0, len(lengths)
        starting_with_empty_page, num_pages, used = False, 0, 0
        while done < n:
            if done == 0 or not starting_with_empty_page:
                # Pages needed for the remaining rows (heap_multi_insert_pages)
                num_pages, avail = 1, BLCKSZ - PAGE_HEADER
                for length in lengths[done:]:
                    size = ITEM_ID + _maxalign(length)
                    if avail < size:
                        num_pages += 1
                        avail = BLCKSZ - PAGE_HEADER
                    avail -= size
                used = 0
            else:
                used += 1
            page = self._page_for(lengths[done], num_pages - used)
            starting_with_empty_page = self.lines[page] == 0
            positions.append(self._put(page, lengths[done]))
            done += 1
            while done < n and self._free_space(page) >= _maxalign(lengths[done]):
                positions.append(self._put(page, lengths[done]))
                done += 1
        return positions


# Rows of a batch of COPY (MAX_BUFFERED_TUPLES and MAX_BUFFERED_BYTES of commands/copyfrom.c)
COPY_BATCH_ROWS = 1000
COPY_BATCH_BYTES = 65535


class CopyLayout:
    """Positions of the rows loaded by a COPY, given the length of their tuples and of their
    line in the input (without its end of line): rows are inserted by batches of
    COPY_BATCH_ROWS rows, or less when their lines reach COPY_BATCH_BYTES bytes."""

    def __init__(self):
        self.heap = HeapLayout(bulk=True)
        self.lengths = []
        self.batch_bytes = 0

    def add(self, length, line_length):
        """Adds a row, and returns the positions of the rows of the batch it completes."""
        self.lengths.append(length)
        self.batch_bytes += line_length
        if len(self.lengths) >= COPY_BATCH_ROWS or self.batch_bytes >= COPY_BATCH_BYTES:
            return self.flush()
        return []

    def flush(self):
        positions = self.heap.multi_insert(self.lengths) if self.lengths else []
        self.lengths, self.batch_bytes = [], 0
        return positions


def _text(value, column):
    """Value of a text column, as stored by Postgres from a Python value."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise TypeError(f"Invalid value {value!r} ({type(value).__name__}) of text column {column}")


def to_arrow(values, column, kind, not_null=False, empty_as_null=False):
    """Arrow array of the Python values of a column of Postgres type `kind`.

    Values are checked as Postgres checks them: integers must fit the type, strings cannot
    contain NUL characters, and `not_null` columns cannot be null. Integers given to text
    columns are stored as strings. With `empty_as_null`, empty strings are null, as with the
    COPY (FORMAT csv) of Postgres."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if kind == VARCHAR:
        try:
            array = pa.array(values, pa.string())
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            array = pa.array([_text(v, column) for v in values], pa.string())
        if pc.any(pc.match_substring(array, "\x00")).as_py():
            raise ValueError(f"Column {column}: strings cannot contain NUL (0x00) characters")
        if empty_as_null:
            array = pc.if_else(pc.equal(array, ""), pa.scalar(None, pa.string()), array)
    elif kind == DOUBLE:
        array = pa.array(values, pa.float64())
    else:
        # Integers are converted after inference, so that floats are refused (instead of
        # being truncated), and values out of the range of the type raise an error
        array = pa.array(values)
        if not (pa.types.is_integer(array.type) or pa.types.is_null(array.type)):
            raise TypeError(f"Column {column} ({kind}) expects integers, got {array.type}")
        target = {BIGINT: pa.int64(), INTEGER: pa.int32(), SMALLINT: pa.int16()}[kind]
        array = array.cast(target)
    if not_null and array.null_count:
        raise ValueError(f"Null value in column {column}, which cannot be null")
    return array


INSERT, COPY = "insert", "copy"


class TableWriter:
    """Appends rows to a staging Parquet file of a table, in order.

    Rows are tuples of the values of `spec.row_columns`, which are converted as Postgres stores
    them when they are loaded by `load`: INSERT (with psycopg2), or COPY (FORMAT csv) of the
    rows written by `csv.writer` (empty strings are null). They are buffered and written by
    batches of `buffer_rows`.

    A writer of the event table can be used in another process than the builder (e.g. to load
    files in parallel): its file is then given to `DatasetBuilder.add_staging_file`. The rows of
    the other tables are written with the length of their heap tuple and of their line in the
    COPY input, from which `DatasetBuilder` finds their position in the Postgres table (see
    `HeapLayout`)."""

    def __init__(self, spec, path, load=INSERT, buffer_rows=1_000_000):
        import pyarrow as pa

        if load not in (INSERT, COPY):
            raise ValueError(f"Unknown load {load}")
        self.spec = spec
        self.path = path
        self.load = load
        self.buffer_rows = buffer_rows
        self.columns = spec.row_columns
        type_of = {VARCHAR: pa.string(), BIGINT: pa.int64(), INTEGER: pa.int32()}
        type_of.update({SMALLINT: pa.int16(), DOUBLE: pa.float64()})
        fields = [(name, type_of[kind]) for name, kind in self.columns]
        self.physical = not spec.is_event_table
        if self.physical:
            fields += [("_tuple_length", pa.int32()), ("_line_length", pa.int32())]
        self.schema = pa.schema(fields)
        self.rows = []
        self.num_rows = 0
        self._writer = None

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.buffer_rows:
            self.flush()

    def add_rows(self, rows):
        for row in rows:
            self.add(row)

    def _line_lengths(self):
        """Lengths in bytes of the lines of COPY (FORMAT csv) of the rows, without end of line."""
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        lengths = []
        for row in self.rows:
            buffer.seek(0)
            buffer.truncate()
            writer.writerow(row)
            lengths.append(len(buffer.getvalue().encode()) - len(writer.dialect.lineterminator))
        return lengths

    def flush(self):
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        width = len(self.columns)
        for row in self.rows:
            if len(row) != width:
                raise ValueError(
                    f"{self.spec.name}: row of {len(row)} values instead of {width}: {row!r}"
                )
        arrays = [
            to_arrow(values, name, kind, name in self.spec.not_null, self.load == COPY)
            for values, (name, kind) in zip(zip(*self.rows), self.columns)
        ]
        if self.physical:
            sizes = heap_tuple_sizes(arrays, [kind for _, kind in self.columns])
            lines = self._line_lengths() if self.load == COPY else [0] * len(self.rows)
            arrays += [pa.array(sizes, pa.int32()), pa.array(lines, pa.int32())]
        if self._writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._writer = pq.ParquetWriter(
                self.path, self.schema, compression="zstd", compression_level=1
            )
        self._writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=self.schema))
        self.num_rows += len(self.rows)
        self.rows = []
        if self.spec.serial:
            check_serial(self.spec, self.num_rows)

    def close(self):
        """Writes the remaining rows. Returns the number of rows of the file (no file is written
        without rows)."""
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.num_rows


def physical_positions(path, load, log=print):
    """Positions (pages, lines) in the Postgres table of the rows of a staging file of a node
    table, in the order of the file (see `HeapLayout`)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["_tuple_length", "_line_length"])
    lengths = table.column("_tuple_length").to_pylist()
    toasted = sum(1 for length in lengths if length > TOAST_TUPLE_THRESHOLD)
    if toasted:
        log(
            f"WARNING: {toasted} rows of {path} are longer than the TOAST threshold of Postgres "
            f"({TOAST_TUPLE_THRESHOLD} bytes), which compresses them: the order of the rows may "
            "differ from the order of the Postgres table"
        )
    if load == INSERT:
        heap = HeapLayout()
        positions = [heap.insert(length) for length in lengths]
    else:
        layout = CopyLayout()
        positions = []
        for length, line in zip(lengths, table.column("_line_length").to_pylist()):
            positions += layout.add(length, line)
        positions += layout.flush()
    pages, lines = zip(*positions)
    return pages, lines


def check_serial(spec, num_rows):
    kind = dict(spec.columns)[spec.serial]
    if num_rows > SERIAL_MAX[kind]:
        raise OverflowError(
            f"{spec.name}: {num_rows} rows, more than the maximum value of its serial column "
            f"{spec.serial} ({kind}, {SERIAL_MAX[kind]})"
        )


class DatasetBuilder:
    """Builds a Parquet dataset from rows given by Python code.

        with DatasetBuilder(dataset_dir, tables, source="...") as builder:
            builder.table("file_node_table").add(row)
            ...

    Rows of each table are written to staging Parquet files as they come (see `TableWriter`),
    so that memory does not depend on the size of the dataset. At the end of the `with` block
    (or with `finish`), node tables are copied in the order of their rows, events are sorted
    (see `write_events_by_day`), and the manifest is written. The dataset is written to
    `<dataset_dir>.tmp`, renamed to `dataset_dir` when complete, and removed on errors.

    Args:
        dataset_dir: folder of the dataset (`<datasets_dir>/<database>`).
        tables: [TableSpec] of the tables of the dataset.
        source: how the dataset was built, recorded in the manifest.
        overwrite: replaces an existing dataset (otherwise, raises FileExistsError).
        load: how the scripts that built the dataset in Postgres loaded it (see `TableWriter`):
            `insert` or `copy`. It determines the null values and the physical order of the node
            tables.
        threads, memory_gb: DuckDB threads and memory limit (default: DuckDB's defaults).
        log: function printing progress messages.
    """

    def __init__(
        self,
        dataset_dir,
        tables,
        source,
        overwrite=False,
        load=INSERT,
        threads=None,
        memory_gb=None,
        row_group_size=ROW_GROUP_SIZE,
        log=print,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        if os.path.exists(self.dataset_dir) and not overwrite:
            raise FileExistsError(
                f"The dataset {self.dataset_dir} already exists. Use --overwrite to replace it."
            )
        self.specs = {spec.name: spec for spec in tables}
        self.source = source
        self.load = load
        self.threads = threads or min(16, os.cpu_count() or 1)
        self.memory_gb = memory_gb
        self.row_group_size = row_group_size
        self.log = log
        self.tmp_dir = self.dataset_dir + ".tmp"
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)
        self.staging_dir = os.path.join(self.tmp_dir, "_staging")
        self._writers = {}
        self._staging = {name: [] for name in self.specs}  # [(path, number of rows)]
        self._done = False

    def table(self, name):
        """TableWriter of a table, whose rows come after those of the files already added."""
        if name not in self._writers:
            self._writers[name] = TableWriter(
                self.specs[name], self.staging_path(name, "main"), self.load
            )
        return self._writers[name]

    def staging_path(self, table, key):
        """Path of a staging file of a table, for a TableWriter of another process."""
        return os.path.join(self.staging_dir, table, f"{key}.parquet")

    def add_staging_file(self, table, path, num_rows):
        """Adds the rows of a staging file of the event table (written by a TableWriter of
        `spec(table)`) after the rows already added to the table."""
        if not self.specs[table].is_event_table:
            raise ValueError(f"{table}: the rows of node tables must be given by `table()`")
        if num_rows:
            self._staging[table].append((path, num_rows))

    def spec(self, table):
        return self.specs[table]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        elif not self._done:
            try:
                self.finish()
            except BaseException:
                self.abort()
                raise
        return False

    def abort(self):
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _query(self, db, spec, files):
        """Query of the rows of a table (with the columns of the table): the rows of node tables
        in the physical order of their Postgres table, events with their serial column."""
        if not files:
            cols = ", ".join(f'CAST(NULL AS {DUCKDB_TYPES[k]}) AS "{n}"' for n, k in spec.columns)
            return f"SELECT {cols} LIMIT 0"
        paths = sql_list([path for path, _ in files])
        import pyarrow as pa

        if not spec.serial:
            # Rows in the physical order of the Postgres table (see `HeapLayout`)
            ((path, num_rows),) = files
            pages, lines = physical_positions(path, self.load, self.log)
            view = f"_positions_{spec.name}"
            positions = {"_row": range(num_rows), "_page": pages, "_line": lines}
            db.register(view, pa.table(positions))
            cols = ", ".join(f's."{n}"' for n in spec.names)
            return (
                f"SELECT {cols} FROM read_parquet({paths}, hive_partitioning = false, "
                f"file_row_number = true) s JOIN {view} p ON s.file_row_number = p._row "
                "ORDER BY p._page, p._line"
            )
        # The serial column numbers the rows in the order of the files and of their rows

        offsets, total = [], 0
        for _, num_rows in files:
            offsets.append(total)
            total += num_rows
        check_serial(spec, total)
        view = f"_offsets_{spec.name}"
        db.register(view, pa.table({"_file": [p for p, _ in files], "_offset": offsets}))
        kind = DUCKDB_TYPES[dict(spec.columns)[spec.serial]]
        cols = ", ".join(
            f'CAST(o._offset + s.file_row_number + 1 AS {kind}) AS "{n}"'
            if n == spec.serial
            else f's."{n}"'
            for n in spec.names
        )
        return (
            f"SELECT {cols} FROM read_parquet({paths}, hive_partitioning = false, "
            f"filename = true, file_row_number = true) s JOIN {view} o ON s.filename = o._file"
        )

    def finish(self):
        """Writes the tables and the manifest, and renames the dataset to its final folder."""
        t0 = time.time()
        for name, writer in self._writers.items():
            # The rows of `table()` come after the files of other processes
            if writer.close():
                self._staging[name].append((writer.path, writer.num_rows))
        duck_tmp = os.path.join(self.tmp_dir, "_duckdb_tmp")
        db = connect_duckdb(duck_tmp, self.threads, self.memory_gb)
        tables = {}
        for name in sorted(self.specs):
            spec, files = self.specs[name], self._staging[name]
            expected = sum(n for _, n in files)
            table_dir = os.path.join(self.tmp_dir, name)
            os.makedirs(table_dir)
            t = time.time()
            query = self._query(db, spec, files)
            if spec.is_event_table:
                if not files:
                    raise ValueError(f"No events in {name}")
                order = EVENT_ORDER + (f', "{spec.serial}"' if spec.serial else "")
                info = write_events_by_day(
                    db,
                    f"({query})",
                    spec.names,
                    os.path.join(self.tmp_dir, f"_staging_{name}"),
                    table_dir,
                    self.row_group_size,
                    lambda msg: self.log(f"{name}: {msg}"),
                    order,
                )
            else:
                path = os.path.join(table_dir, "part-00000.parquet")
                copy_to_parquet(db, query, path, self.row_group_size)
                info = {"files": ["part-00000.parquet"], "order": "physical"}
            num_rows = count_rows(db, [os.path.join(table_dir, f) for f in info["files"]])
            if num_rows != expected:
                raise RuntimeError(f"{name}: {num_rows} rows written, {expected} given")
            info.update(num_rows=num_rows, columns=[list(c) for c in spec.columns])
            tables[name] = info
            for path, _ in files:
                os.remove(path)
            self.log(
                f"{name}: {num_rows:,} rows, {len(info['files'])} files ({time.time() - t:.0f}s)"
            )
        db.close()
        shutil.rmtree(duck_tmp, ignore_errors=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        write_manifest(self.tmp_dir, os.path.basename(self.dataset_dir), self.source, tables)
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
        os.rename(self.tmp_dir, self.dataset_dir)
        self._done = True
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, names in os.walk(self.dataset_dir)
            for f in names
        )
        self.log(
            f"Dataset written to {self.dataset_dir}: {size / 1e9:.2f} GB "
            f"({(time.time() - t0) / 60:.1f} min to sort and write the tables)"
        )


def update_table(dataset_dir, table, column, value, where, relations=None, note=None, log=print):
    """`UPDATE table SET column = value WHERE where` on a Parquet dataset: rewrites the files of
    the table in place (rows keep their order). `relations` ({name: Arrow table}) are tables
    that `where` may query. `note` is added to the source of the dataset in its manifest.
    Returns the number of rows matched by `where`."""
    with open(os.path.join(dataset_dir, "manifest.json")) as f:
        manifest = json.load(f)
    info = manifest["tables"][table]
    db = connect_duckdb(os.path.join(dataset_dir, "_duckdb_tmp"))
    for name, relation in (relations or {}).items():
        db.register(name, relation)
    cols = ", ".join(
        f'CAST(CASE WHEN {where} THEN {value} ELSE "{name}" END AS {DUCKDB_TYPES[kind]}) '
        f'AS "{name}"'
        if name == column
        else f'"{name}"'
        for name, kind in info["columns"]
    )
    updated = 0
    for name in info["files"]:
        path = os.path.join(dataset_dir, table, name)
        rows = f"read_parquet({sql_str(path)}, file_row_number = true)"
        n = db.execute(f"SELECT count(*) FROM {rows} WHERE {where};").fetchone()[0]
        if n:
            tmp = path + ".tmp"
            # `where` may be a join, which does not keep the order of the rows
            copy_to_parquet(db, f"SELECT {cols} FROM {rows} ORDER BY file_row_number", tmp)
            if count_rows(db, [tmp]) != count_rows(db, [path]):
                raise RuntimeError(f"{tmp}: wrong number of rows")
            os.replace(tmp, path)
            log(f"{table}/{name}: {n} rows updated")
        updated += n
    db.close()
    shutil.rmtree(os.path.join(dataset_dir, "_duckdb_tmp"), ignore_errors=True)
    if note and note not in manifest["source"]:
        manifest["source"] += f", {note}"
        with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    return updated
