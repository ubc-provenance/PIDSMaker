"""Database backends: a Postgres server, or Parquet files queried in-process by DuckDB.

Both are opened by `init_database_connection(cfg)`, which returns (cursor, connection). The
Parquet backend exposes the subset of the psycopg2 API used by the framework (`cursor(name)`,
`execute(sql, params)` with `%s` placeholders, `fetchone`/`fetchmany`/`fetchall`, `commit`,
`close`, context managers), so that queries run unchanged on both backends.

A Parquet dataset is a folder `<datasets_dir>/<database>/` with one subfolder per table,
holding the `part-*.parquet` files of the table, and a `manifest.json` (see
`dataset_preprocessing/export_parquet.py`). Each table is exposed as a DuckDB view with the
name and columns of the Postgres table.

Row order. Rows are stored in the order of the Postgres queries of the framework, so that
results are identical:
- node tables in their physical order (the order of `SELECT ... FROM table` in Postgres);
- the event table sorted by `timestamp_rec, event_uuid`, sorted by Postgres (with the collation
  of the database, which differs from the byte order of DuckDB for some strings).
DuckDB returns the rows of queries without ORDER BY in their stored order
(`preserve_insertion_order`, enabled by default), so event queries skip the sort on this
backend (see `events_order_sql`).
"""

import json
import os
import tempfile

POSTGRES = "postgres"
PARQUET = "parquet"
BACKENDS = (POSTGRES, PARQUET)

DEFAULT_DATASETS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pidsmaker", "datasets")

# Threads of each DuckDB connection: construction opens one connection per day, several at
# the same time (prefetcher threads or forked workers)
DUCKDB_THREADS = int(os.environ.get("PIDSMAKER_DUCKDB_THREADS", 8))


def database_name(cfg):
    return cfg.dataset.database_all_file if cfg.construction.use_all_files else cfg.dataset.database


def database_backend(cfg):
    return getattr(cfg.database, "backend", None) or PARQUET


def parquet_dataset_dir(cfg):
    datasets_dir = getattr(cfg.database, "datasets_dir", None) or DEFAULT_DATASETS_DIR
    return os.path.join(datasets_dir, database_name(cfg))


def init_database_connection(cfg):
    """(cursor, connection) of the database of the dataset of `cfg`, on its backend."""
    backend = database_backend(cfg)
    if backend == PARQUET:
        from pidsmaker.datasets import ensure_downloaded

        ensure_downloaded(cfg)
        connect = ParquetConnection(parquet_dataset_dir(cfg))
        return connect.cursor(), connect
    if backend == POSTGRES:
        return _init_postgres_connection(cfg)
    raise ValueError(f"Unknown database backend {backend}. Available backends: {BACKENDS}")


def _init_postgres_connection(cfg):
    import psycopg2

    # Queries without ORDER BY (e.g. `select * from file_node_table`) return rows in the
    # table's physical order, which determines e.g. the order of indexid2msg and thus the
    # featurization. With synchronized seqscans (Postgres default), a scan started while
    # another scan of the same table is running joins it midway and returns rotated rows,
    # making results depend on concurrent runs. Disabled for this session only.
    kwargs = dict(
        database=database_name(cfg),
        user=cfg.database.user,
        password=cfg.database.password,
        port=cfg.database.port,
        options="-c synchronize_seqscans=off",
    )
    if cfg.database.host is not None:
        kwargs["host"] = cfg.database.host
    connect = psycopg2.connect(**kwargs)
    return connect.cursor(), connect


def events_order_sql(connect, columns="timestamp_rec, event_uuid"):
    """ORDER BY clause of the event queries sorted by time, run on `connect` (a connection or
    a cursor).

    Empty on the Parquet backend, whose event table is stored in this order (by Postgres, see
    the module docstring). Only `timestamp_rec, event_uuid` is a total order: queries sorted
    by `timestamp_rec` alone return the events of a same timestamp in an order that Postgres
    does not define, which is the `event_uuid` order on the Parquet backend.
    """
    if getattr(connect, "events_presorted", False):
        return ""
    return f"ORDER BY {columns}"


class ParquetConnection:
    """psycopg2-like connection to a Parquet dataset, through an in-memory DuckDB database."""

    events_presorted = True

    def __init__(self, dataset_dir):
        import duckdb

        manifest_path = os.path.join(dataset_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No Parquet dataset in {dataset_dir} (missing manifest.json). Download it with "
                f"`pidsmaker download {os.path.basename(dataset_dir)} --datasets_dir "
                f"{os.path.dirname(dataset_dir)}` (see pidsmaker/datasets.py)."
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.dataset_dir = dataset_dir
        # Queries stream their results and do not sort events: DuckDB only spills to disk (in
        # the system's temporary directory, not the working directory) for large sorts
        self._db = duckdb.connect(
            config={
                "threads": DUCKDB_THREADS,
                "temp_directory": os.path.join(tempfile.gettempdir(), "pidsmaker_duckdb"),
            }
        )
        for table, info in self.manifest["tables"].items():
            files = [os.path.join(dataset_dir, table, name) for name in info["files"]]
            file_list = ", ".join("'" + p.replace("'", "''") + "'" for p in files)
            self._db.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{file_list}]);")

    def cursor(self, name=None):
        # Named (server-side) cursors of psycopg2 stream their rows: all cursors stream here
        return ParquetCursor(self._db.cursor())

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def closed(self):
        return self._db is None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _to_duckdb_sql(sql):
    """psycopg2 placeholders (`%s`, with `%%` for `%`) to DuckDB ones (`?`)."""
    return sql.replace("%s", "?").replace("%%", "%")


class ParquetCursor:
    """psycopg2-like cursor. Rows are tuples of Python values, as with psycopg2.

    DuckDB streams the results of `fetchmany` (the result is not materialized at `execute`),
    as the named (server-side) cursors of psycopg2."""

    events_presorted = True

    def __init__(self, cursor):
        self._cur = cursor

    def execute(self, sql, params=None):
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(_to_duckdb_sql(sql), list(params))
        return self

    def fetchmany(self, size=1):
        return self._cur.fetchmany(size)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        while True:
            rows = self._cur.fetchmany(10_000)
            if not rows:
                return
            yield from rows

    def close(self):
        self._cur.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
