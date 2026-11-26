"""Datasets: Parquet files queried in-process by DuckDB.

`init_database_connection(cfg)` returns (cursor, connection) for the dataset of `cfg`. The
connection exposes the DB-API subset used by the framework (`cursor(name)`, `execute(sql,
params)` with `%s` placeholders, `fetchone`/`fetchmany`/`fetchall`, `commit`, `close`, context
managers).

A dataset is a folder `<datasets_dir>/<database>/` with one subfolder per table, holding the
`part-*.parquet` files of the table, and a `manifest.json` listing them (see
`pidsmaker/utils/dataset_writer.py`). Each table is exposed as a DuckDB view named after it.

Row order. Rows are stored in the order in which the framework reads them, which determines
e.g. the order of the corpora of featurization: node tables in the order of the dataset, and the
event table sorted by `timestamp_rec, event_uuid`. DuckDB returns the rows of queries without
ORDER BY in their stored order (`preserve_insertion_order`, enabled by default), so the event
queries need no ORDER BY.
"""

import json
import os
import tempfile

DEFAULT_DATASETS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "pidsmaker", "datasets")

# Threads of each DuckDB connection: construction opens one connection per day, several at
# the same time (prefetcher threads or forked workers)
DUCKDB_THREADS = int(os.environ.get("PIDSMAKER_DUCKDB_THREADS", 8))


def database_name(cfg):
    return cfg.dataset.database_all_file if cfg.construction.use_all_files else cfg.dataset.database


def datasets_dir(cfg):
    return getattr(cfg.database, "datasets_dir", None) or DEFAULT_DATASETS_DIR


def parquet_dataset_dir(cfg):
    return os.path.join(datasets_dir(cfg), database_name(cfg))


def init_database_connection(cfg):
    """(cursor, connection) of the dataset of `cfg`, downloaded first if it is missing."""
    from pidsmaker.datasets import ensure_downloaded

    ensure_downloaded(cfg)
    connect = ParquetConnection(parquet_dataset_dir(cfg))
    return connect.cursor(), connect


class ParquetConnection:
    """DB-API-like connection to a Parquet dataset, through an in-memory DuckDB database."""

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
        # `name` is accepted for the callers that name their cursors: all cursors stream
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
    """`%s` placeholders (with `%%` for `%`), as written in the queries, to DuckDB ones (`?`)."""
    return sql.replace("%s", "?").replace("%%", "%")


class ParquetCursor:
    """DB-API-like cursor. Rows are tuples of Python values.

    DuckDB streams the results of `fetchmany`: the result is not materialized at `execute`."""

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
