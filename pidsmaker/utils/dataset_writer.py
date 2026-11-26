"""Writing of Parquet datasets (format: see `pidsmaker/utils/database.py`), used by the scripts
that build datasets from their raw files (`dataset_preprocessing/`), which give the rows of each
table from Python (`DatasetBuilder`):

- a node table is a single file, whose rows are in the order in which they were given;
- the event table is sorted by `timestamp_rec, event_uuid` (then `_id`, for duplicate records),
  one file per day (UTC);
- the serial column `_id` of event tables numbers the events from 1 in the order in which they
  are given.

The tables have the columns, with their names, order and types, of the published datasets (see
`TableSpec`), and values follow the same rules (see `to_arrow`).
"""

import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass

EVENT_ORDER = "timestamp_rec, event_uuid"
DAY_NS = 86_400 * 10**9
ROW_GROUP_SIZE = 1_000_000

# SQL types of the columns of the datasets (as named in the manifests), and their DuckDB types
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
# Largest value of a serial column of each type
SERIAL_MAX = {INTEGER: 2**31 - 1, BIGINT: 2**63 - 1}


def sql_str(s):
    return "'" + s.replace("'", "''") + "'"


def sql_list(paths):
    return "[" + ", ".join(sql_str(p) for p in paths) + "]"


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

    # Partition folders are `_day=<day>`, NULL timestamps (if any) in a non-numeric one, sorted
    # last
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
    return {"files": files, "order": f"ORDER BY {EVENT_ORDER}"}


def write_manifest(dataset_dir, database, source, tables):
    """Writes the manifest of a dataset. `tables` is {table: description} (files, order, number
    of rows and columns of each table)."""
    manifest = {
        "format_version": 1,
        "database": database,
        "source": source,
        "created_at": utc_now(),
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
        columns: [(column, SQL type)] in the order of the table.
        not_null: columns that cannot be null.
        serial: column numbering the rows from 1 in the order in which they are given (filled
            by the writer, not given in rows), or None.
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
    the label column. The `_id` of events is an `integer` in the provenance datasets, a
    `bigint` in the network datasets."""
    tables = [node_table_spec(nt.table, [col for _, col in nt.columns]) for nt in schema.node_types]
    extra = [
        (a.column, VARCHAR if a.kind == "categorical" else DOUBLE) for a in schema.edge_attributes
    ]
    if schema.label_column:
        extra.append((schema.label_column, SMALLINT))
    tables.append(event_table_spec(schema.event_table, extra, serial_type))
    return tables


def _text(value, column):
    """Value of a text column from a Python value: integers are stored as strings."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise TypeError(f"Invalid value {value!r} ({type(value).__name__}) of text column {column}")


def to_arrow(values, column, kind, not_null=False, empty_as_null=False):
    """Arrow array of the Python values of a column of SQL type `kind`.

    Integers must fit the type (floats are refused), and `not_null` columns cannot be null.
    Integers given to text columns are stored as strings. With `empty_as_null`, empty strings
    are null (as in the network datasets)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if kind == VARCHAR:
        try:
            array = pa.array(values, pa.string())
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            array = pa.array([_text(v, column) for v in values], pa.string())
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


class TableWriter:
    """Appends rows to a staging Parquet file of a table, in order.

    Rows are tuples of the values of `spec.row_columns`, converted by `to_arrow` (with
    `empty_as_null`, empty strings are null). They are buffered and written by batches of
    `buffer_rows`.

    A writer of the event table can be used in another process than the builder (e.g. to load
    files in parallel): its file is then given to `DatasetBuilder.add_staging_file`."""

    def __init__(self, spec, path, empty_as_null=False, buffer_rows=1_000_000):
        import pyarrow as pa

        self.spec = spec
        self.path = path
        self.empty_as_null = empty_as_null
        self.buffer_rows = buffer_rows
        self.columns = spec.row_columns
        type_of = {VARCHAR: pa.string(), BIGINT: pa.int64(), INTEGER: pa.int32()}
        type_of.update({SMALLINT: pa.int16(), DOUBLE: pa.float64()})
        fields = [(name, type_of[kind]) for name, kind in self.columns]
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
            to_arrow(values, name, kind, name in self.spec.not_null, self.empty_as_null)
            for values, (name, kind) in zip(zip(*self.rows), self.columns)
        ]
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
    (or with `finish`), node tables are copied in the order of their rows, events are numbered
    and sorted (see `write_events_by_day`), and the manifest is written. The dataset is written to
    `<dataset_dir>.tmp`, renamed to `dataset_dir` when complete, and removed on errors.

    Args:
        dataset_dir: folder of the dataset (`<datasets_dir>/<database>`).
        tables: [TableSpec] of the tables of the dataset.
        source: how the dataset was built, recorded in the manifest.
        overwrite: replaces an existing dataset (otherwise, raises FileExistsError).
        empty_as_null: empty strings are stored as null (see `to_arrow`).
        threads, memory_gb: DuckDB threads and memory limit (default: DuckDB's defaults).
        log: function printing progress messages.
    """

    def __init__(
        self,
        dataset_dir,
        tables,
        source,
        overwrite=False,
        empty_as_null=False,
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
        self.empty_as_null = empty_as_null
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
                self.specs[name], self.staging_path(name, "main"), self.empty_as_null
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
        """Query of the rows of a table (with the columns of the table), in the order of its
        files and of their rows, with the serial column of events."""
        if not files:
            cols = ", ".join(f'CAST(NULL AS {DUCKDB_TYPES[k]}) AS "{n}"' for n, k in spec.columns)
            return f"SELECT {cols} LIMIT 0"
        paths = sql_list([path for path, _ in files])
        if not spec.serial:
            # DuckDB keeps the order of the files and of their rows (`preserve_insertion_order`)
            cols = ", ".join(f'"{n}"' for n in spec.names)
            return f"SELECT {cols} FROM read_parquet({paths}, hive_partitioning = false)"
        import pyarrow as pa

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
                info = {"files": ["part-00000.parquet"], "order": "insertion"}
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
