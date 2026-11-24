"""Extracts the small sample of CLEARSCOPE_E3 used by the tests (`tests/fixtures/clearscope_e3_sample/`).

In test mode, the pipeline only builds the first time window graph (truncated to a few thousand
edges) of the first day of the train, val and test sets. The sample therefore keeps, for each of
these days, the first `--events_per_day` events, the nodes they touch and the ground truth nodes.
On the test day, the sample starts at the beginning of the attack instead of midnight, so that
the test graph contains malicious nodes. The sample also keeps the events of the attack time windows
between two ground truth nodes: they are the malicious edges of the edge-level evaluation with
`malicious_edge_selection=both_nodes` (e.g. euler, jbeil), which are read from the database.

Tables are exported with the CSV format of Postgres' COPY (which distinguishes NULL from empty
strings) and loaded by `tests/conftest.py`, with the tables of `schema.sql`.

Usage (requires the full CLEARSCOPE_E3 database):
    python tests/fixtures/make_clearscope_e3_sample.py [--database_host postgres] [--events_per_day 5000]
"""

import argparse
import csv
import gzip
import io
import os
import sys
from datetime import datetime

import psycopg2
import pytz

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pidsmaker.config import DATASET_DEFAULT_CONFIG  # noqa: E402

DATASET = "CLEARSCOPE_E3"
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clearscope_e3_sample")
GROUND_TRUTH_DIR = os.path.join(ROOT, "Ground_Truth", "orthrus")
TABLE_COLUMNS = {
    "event_table": [
        "src_node",
        "src_index_id",
        "operation",
        "dst_node",
        "dst_index_id",
        "event_uuid",
        "timestamp_rec",
        "_id",
    ],
    "subject_node_table": ["node_uuid", "hash_id", "path", "cmd", "index_id"],
    "file_node_table": ["node_uuid", "hash_id", "path", "index_id"],
    "netflow_node_table": [
        "node_uuid",
        "hash_id",
        "src_addr",
        "src_port",
        "dst_addr",
        "dst_port",
        "index_id",
    ],
}


def to_ns(date_str, tz):
    return int(tz.localize(datetime.fromisoformat(date_str)).timestamp() * 1e9)


def day_ranges(dataset_cfg):
    """(day, start, end) in ns of the events kept for the first day of each set."""
    tz = pytz.timezone(dataset_cfg["timezone"])
    attack_starts = {start[:10]: start for _, start, _ in dataset_cfg["attack_to_time_window"]}
    days = [dataset_cfg[f"{split}_dates"][0] for split in ["train", "val", "test"]]
    return [
        (day, to_ns(attack_starts.get(day, day), tz), to_ns(day, tz) + 86400 * 10**9)
        for day in days
    ]


def attack_windows(dataset_cfg):
    """(start, end) in ns of the attacks."""
    tz = pytz.timezone(dataset_cfg["timezone"])
    return [(to_ns(start, tz), to_ns(end, tz)) for _, start, end in dataset_cfg["attack_to_time_window"]]


def ground_truth_uuids(dataset_cfg):
    uuids = []
    for rel_path in dataset_cfg["ground_truth_relative_path"]:
        with open(os.path.join(GROUND_TRUTH_DIR, rel_path)) as f:
            uuids += [row[0].strip() for row in csv.reader(f) if row]
    return uuids


def export(cur, table, query, params):
    """Writes the rows of `query` to `<table>.csv.gz` with the CSV format of COPY."""
    buf = io.StringIO()
    cur.copy_expert(
        f"COPY ({cur.mogrify(query, params).decode()}) TO STDOUT WITH (FORMAT csv, HEADER true)",
        buf,
    )
    # mtime=0 keeps the files identical when the sample is extracted again
    with open(os.path.join(SAMPLE_DIR, f"{table}.csv.gz"), "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(buf.getvalue().encode())
    print(f"{table}: {buf.getvalue().count(chr(10)) - 1} rows")


def main(args):
    dataset_cfg = DATASET_DEFAULT_CONFIG[DATASET]
    connect = psycopg2.connect(
        database=args.database or dataset_cfg["database"],
        host=args.database_host,
        user=args.database_user,
        password=args.database_password,
        port=args.database_port,
    )
    cur = connect.cursor()
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    gt_uuids = ground_truth_uuids(dataset_cfg)
    cur.execute(
        " UNION ".join(
            f"SELECT index_id::text FROM {table} WHERE node_uuid = ANY(%s)"
            for table in TABLE_COLUMNS
            if table != "event_table"
        ),
        [gt_uuids] * (len(TABLE_COLUMNS) - 1),
    )
    gt_node_ids = sorted(row[0] for row in cur.fetchall())

    columns = ", ".join(TABLE_COLUMNS["event_table"])
    day_query = (
        f"(SELECT {columns} FROM event_table WHERE timestamp_rec >= %s AND timestamp_rec < %s "
        "ORDER BY timestamp_rec, event_uuid LIMIT %s)"
    )
    attack_query = (
        f"(SELECT {columns} FROM event_table WHERE timestamp_rec BETWEEN %s AND %s "
        "AND src_index_id::text = ANY(%s) AND dst_index_id::text = ANY(%s))"
    )
    ranges = day_ranges(dataset_cfg)
    windows = attack_windows(dataset_cfg)
    events_query = (
        "SELECT * FROM ("
        + " UNION ".join([day_query] * len(ranges) + [attack_query] * len(windows))
        + ") AS events ORDER BY timestamp_rec, event_uuid"
    )
    params = [p for _, start, end in ranges for p in (start, end, args.events_per_day)]
    params += [p for start, end in windows for p in (start, end, gt_node_ids, gt_node_ids)]
    export(cur, "event_table", events_query, params)

    cur.execute(f"SELECT src_index_id, dst_index_id FROM ({events_query}) AS events;", params)
    node_ids = sorted({str(node_id) for row in cur.fetchall() for node_id in row})
    for table, columns in TABLE_COLUMNS.items():
        if table == "event_table":
            continue
        query = (
            f"SELECT {', '.join(columns)} FROM {table} "
            "WHERE index_id::text = ANY(%s) OR node_uuid = ANY(%s) ORDER BY index_id"
        )
        export(cur, table, query, (node_ids, gt_uuids))
    connect.close()
    print(f"Sample written to {SAMPLE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database", default=None, help="Source database (default: that of CLEARSCOPE_E3)"
    )
    parser.add_argument("--database_host", default="postgres")
    parser.add_argument("--database_user", default="postgres")
    parser.add_argument("--database_password", default="postgres")
    parser.add_argument("--database_port", type=int, default=5432)
    parser.add_argument("--events_per_day", type=int, default=5000)
    main(parser.parse_args())
