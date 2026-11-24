"""Helpers of the preprocessing scripts of datasets whose tables are defined by their schema
(e.g. LANL, OpTC network flows): database and table creation, and bulk loading."""

import csv
import io

import psycopg2

from pidsmaker.utils.dataset_utils import create_schema_tables, get_schema
from pidsmaker.utils.utils import init_database_connection, log


def get_database_name(cfg):
    return cfg.dataset.database_all_file if cfg.construction.use_all_files else cfg.dataset.database


def ensure_database(cfg):
    """Creates the database of the dataset if it doesn't exist."""
    database = get_database_name(cfg)
    connect = psycopg2.connect(
        dbname="postgres",
        host=cfg.database.host,
        user=cfg.database.user,
        password=cfg.database.password,
        port=cfg.database.port,
    )
    connect.autocommit = True
    cur = connect.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (database,))
    if cur.fetchone() is None:
        # template0 avoids failures when template1 has a collation version mismatch
        # (e.g. after an OS upgrade of the Postgres container)
        cur.execute(f'CREATE DATABASE "{database}" TEMPLATE template0;')
        log(f"Created database {database}")
    connect.close()


def create_tables(cfg, overwrite):
    """Creates the tables of the dataset's schema. Existing tables are dropped with `overwrite`."""
    schema = get_schema(cfg)
    cur, connect = init_database_connection(cfg)
    cur.execute(f"SELECT to_regclass('{schema.event_table}');")
    if cur.fetchone()[0] is not None:
        if not overwrite:
            raise RuntimeError(
                f"The tables of {get_database_name(cfg)} already exist. Use --overwrite to recreate them."
            )
        for table in [nt.table for nt in schema.node_types] + [schema.event_table]:
            cur.execute(f"DROP TABLE IF EXISTS {table};")
    create_schema_tables(schema, cur)
    connect.commit()
    connect.close()


def copy_rows(cur, table, columns, rows):
    """Bulk inserts rows (iterable of tuples) with COPY."""
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    buffer.seek(0)
    cur.copy_expert(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)", buffer)


def index_events_by_time(cfg):
    """Index used by the construction, which queries events by time range."""
    schema = get_schema(cfg)
    cur, connect = init_database_connection(cfg)
    log("Indexing events by time...")
    cur.execute(f"CREATE INDEX IF NOT EXISTS event_time_idx ON {schema.event_table} (timestamp_rec);")
    if schema.label_column:
        # The labelling queries the few malicious events, which would otherwise scan the table
        log("Indexing malicious events...")
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS event_label_idx ON {schema.event_table} (timestamp_rec) "
            f"WHERE {schema.label_column} = 1;"
        )
    connect.commit()
    connect.close()
