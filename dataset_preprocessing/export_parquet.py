"""Exports Postgres databases of datasets to Parquet datasets (see `pidsmaker/utils/database.py`).

    python dataset_preprocessing/export_parquet.py clearscope_e3 cadets_e3 --out_dir ~/.cache/pidsmaker/datasets

Writes `<out_dir>/<database>/<table>/part-*.parquet` and `<out_dir>/<database>/manifest.json`.
Tables are read by DuckDB (postgres extension), with the column names, order and types of
Postgres. Rows keep the order of the queries of the framework on Postgres, so that both backends
give identical results:

- node tables keep their physical order, the order of their scans (`SELECT ... FROM table`);
- the event table is sorted by `timestamp_rec, event_uuid`. It is not sorted by Postgres, whose
  sort of the largest tables needs hundreds of GB of temporary files: events are copied unsorted,
  partitioned by day (UTC), and each day is sorted by DuckDB. DuckDB compares strings byte by
  byte, Postgres with the collation of the database (e.g. en_US.utf8, which ignores dashes and
  case at first). Both orders are the same when all the event UUIDs are canonical UUIDs of a
  single case, or alphanumeric strings of a single case, or more generally when no letter has
  both cases and the events of each timestamp have IDs of the same shape (same length and
  separators at the same places, e.g. `agraph_12_0000001` in LANL): this is checked (`tie_order`
  of the manifest). One file is written per day. Datasets that fail this check can be exported
  with `--sort_in_postgres`, which sorts their events in Postgres into a single file.

Exports are written to `<out_dir>/<database>.tmp` and renamed when complete. Row counts are
checked against Postgres. The files are written as by the scripts that build datasets from their
raw files (see `pidsmaker/utils/dataset_writer.py`).
"""

import argparse
import datetime
import json
import os
import shutil
import time

import duckdb

from pidsmaker.utils.dataset_writer import (
    EVENT_ORDER,
    check_tie_order,
    connect_duckdb,
    copy_to_parquet,
    count_rows,
    free_gb,
    sql_str,
    utc_now,
    write_events_by_day,
    write_manifest,
)


def log(database, msg):
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} [{database}] {msg}", flush=True)


def pg_query(db, sql):
    """Runs `sql` on Postgres and returns its rows."""
    return db.execute(f"SELECT * FROM postgres_query('pg', {sql_str(sql)});").fetchall()


def export_database(database, args):
    out_dir = os.path.join(args.out_dir, database)
    tmp_dir = out_dir + ".tmp"
    if os.path.exists(out_dir):
        if not args.overwrite:
            log(database, f"{out_dir} exists, skipped (--overwrite to replace it)")
            return
        shutil.rmtree(out_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    t0 = time.time()

    duck_tmp = os.path.join(tmp_dir, "_duckdb_tmp")
    db = connect_duckdb(duck_tmp, args.threads, args.memory_gb)
    db.execute("INSTALL postgres; LOAD postgres;")
    # Scans of node tables must start at the first page and not be parallel, for their
    # physical order (as in `pidsmaker/utils/database.py`)
    options = "-c synchronize_seqscans=off -c max_parallel_workers_per_gather=0"
    dsn = (
        f"host={args.database_host} port={args.database_port} dbname={database} "
        f"user={args.database_user} password={args.database_password} options='{options}'"
    )
    db.execute(f"ATTACH {sql_str(dsn)} AS pg (TYPE postgres, READ_ONLY);")

    db_size = pg_query(db, "SELECT pg_database_size(current_database())")[0][0] / 1e9
    collation = pg_query(
        db, "SELECT datcollate FROM pg_database WHERE datname = current_database()"
    )[0][0]
    tables = [
        row[0]
        for row in pg_query(
            db,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' "
            "AND table_type = 'BASE TABLE' ORDER BY table_name",
        )
    ]
    # Staging (unsorted events) and output each take about a fifth of the Postgres size at most
    needed = 2 * db_size / 5 + args.min_free_gb
    if free_gb(args.out_dir) < needed:
        raise RuntimeError(
            f"{free_gb(args.out_dir):.0f} GB free, {needed:.0f} GB needed for {database} "
            f"({db_size:.0f} GB in Postgres, with a margin of {args.min_free_gb} GB)"
        )
    log(database, f"{db_size:.1f} GB in Postgres, tables {tables}")

    exported_at = utc_now()
    manifest_tables = {}
    for table in tables:
        columns = pg_query(
            db,
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = "
            f"'public' AND table_name = '{table}' ORDER BY ordinal_position",
        )
        names = [name for name, _ in columns]
        cols = ", ".join(f'"{n}"' for n in names)
        is_event_table = "timestamp_rec" in names and "event_uuid" in names
        table_dir = os.path.join(tmp_dir, table)
        os.makedirs(table_dir)
        t = time.time()
        if is_event_table and args.sort_in_postgres:
            query = f"SELECT {cols} FROM {table} ORDER BY {EVENT_ORDER}"
            copy_to_parquet(
                db,
                f"SELECT * FROM postgres_query('pg', {sql_str(query)})",
                os.path.join(table_dir, "part-00000.parquet"),
                args.row_group_size,
            )
            info = {
                "files": ["part-00000.parquet"],
                "order": f"ORDER BY {EVENT_ORDER}",
                "tie_order": "postgres",
            }
        elif is_event_table:
            info = write_events_by_day(
                db,
                f"pg.public.{table}",
                names,
                os.path.join(tmp_dir, f"_staging_{table}"),
                table_dir,
                args.row_group_size,
                lambda msg: log(database, f"{table}: {msg}"),
            )
            if info["tie_order"] == "unverified":
                log(database, f"{table}: see --sort_in_postgres to sort the events in Postgres")
        else:
            copy_to_parquet(
                db,
                f"SELECT * FROM postgres_query('pg', {sql_str(f'SELECT {cols} FROM {table}')})",
                os.path.join(table_dir, "part-00000.parquet"),
                args.row_group_size,
            )
            info = {"files": ["part-00000.parquet"], "order": "physical"}

        num_rows = count_rows(db, [os.path.join(table_dir, f) for f in info["files"]])
        pg_rows = pg_query(db, f"SELECT count(*) FROM {table}")[0][0]
        if num_rows != pg_rows:
            raise RuntimeError(f"{table}: {num_rows} rows exported, {pg_rows} in Postgres")
        info.update(num_rows=num_rows, columns=[list(c) for c in columns])
        manifest_tables[table] = info
        log(
            database,
            f"{table}: {num_rows:,} rows, {len(info['files'])} files ({time.time() - t:.0f}s)",
        )

    db.close()
    shutil.rmtree(duck_tmp, ignore_errors=True)
    write_manifest(tmp_dir, database, "postgres", manifest_tables, collation, exported_at)
    os.rename(tmp_dir, out_dir)
    size = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(out_dir)
        for name in names
    )
    log(
        database,
        f"done: {size / 1e9:.2f} GB ({db_size / max(size / 1e9, 1e-9):.1f}x smaller than Postgres) "
        f"in {(time.time() - t0) / 60:.1f} min",
    )


def recheck(dataset_dir):
    """Checks the tie order of the event table of an existing export, updating its manifest."""
    path = os.path.join(dataset_dir, "manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    db = duckdb.connect()
    for table, info in manifest["tables"].items():
        if "tie_order" not in info or info["tie_order"] == "postgres":
            continue
        files = ", ".join(sql_str(os.path.join(dataset_dir, table, f)) for f in info["files"])
        info["tie_order"] = check_tie_order(
            db, f"read_parquet([{files}])", lambda msg: log(manifest["database"], msg)
        )
        log(manifest["database"], f"{table}: tie order {info['tie_order']}")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("databases", nargs="+", help="Postgres databases (e.g. clearscope_e3)")
    parser.add_argument("--out_dir", required=True, help="Folder of the Parquet datasets")
    parser.add_argument("--database_host", default="postgres")
    parser.add_argument("--database_user", default="postgres")
    parser.add_argument("--database_password", default="postgres")
    parser.add_argument("--database_port", type=int, default=5432)
    parser.add_argument("--threads", type=int, default=16, help="DuckDB threads")
    parser.add_argument("--memory_gb", type=int, default=150, help="DuckDB memory limit")
    parser.add_argument("--row_group_size", type=int, default=1_000_000)
    parser.add_argument(
        "--min_free_gb", type=int, default=100, help="Disk space left free (checked at start)"
    )
    parser.add_argument(
        "--sort_in_postgres",
        action="store_true",
        help="Sort the events in Postgres (exact order whatever their UUIDs, but needs temporary "
        "files of the size of the event table on the Postgres server), into a single file",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="Only check again the tie order of existing exports (and update their manifest)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for database in args.databases:
        if args.recheck:
            recheck(os.path.join(args.out_dir, database))
        else:
            export_database(database, args)


if __name__ == "__main__":
    main()
