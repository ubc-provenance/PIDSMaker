"""Extracts the small sample of CLEARSCOPE_E3 used by the tests (`tests/fixtures/clearscope_e3_sample/`).

In test mode, the pipeline only builds the first time window graph (truncated to a few thousand
edges) of the first day of the train, val and test sets. The sample therefore keeps, for each of
these days, the first `--events_per_day` events, the nodes they touch and the ground truth nodes.
On the test day, the sample starts at the beginning of the attack instead of midnight, so that
the test graph contains malicious nodes. The sample also keeps the events of the attack time windows
between two ground truth nodes: they are the malicious edges of the edge-level evaluation with
`malicious_edge_selection=both_nodes` (e.g. euler, jbeil), which are read from the dataset.

The sample is a dataset (see `pidsmaker/utils/database.py`), read by `tests/conftest.py`: events
sorted by `timestamp_rec, event_uuid`, nodes sorted by `index_id`.

Usage (reads the full CLEARSCOPE_E3 dataset, downloaded first if it is missing):
    python tests/fixtures/make_clearscope_e3_sample.py [--datasets_dir DIR] [--events_per_day 5000]
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime

import duckdb
import pytz

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pidsmaker.config import DATASET_DEFAULT_CONFIG  # noqa: E402
from pidsmaker.datasets import download  # noqa: E402

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
    return [
        (to_ns(start, tz), to_ns(end, tz)) for _, start, end in dataset_cfg["attack_to_time_window"]
    ]


def ground_truth_uuids(dataset_cfg):
    uuids = []
    for rel_path in dataset_cfg["ground_truth_relative_path"]:
        with open(os.path.join(GROUND_TRUTH_DIR, rel_path)) as f:
            uuids += [row[0].strip() for row in csv.reader(f) if row]
    return uuids


def sql_list(values):
    return "(" + ", ".join("'" + str(v).replace("'", "''") + "'" for v in values) + ")"


def main(args):
    dataset_cfg = DATASET_DEFAULT_CONFIG[DATASET]
    (source,) = download(DATASET, datasets_dir=args.datasets_dir)
    with open(os.path.join(source, "manifest.json")) as f:
        source_tables = json.load(f)["tables"]
    db = duckdb.connect()
    for table, info in source_tables.items():
        files = sql_list(os.path.join(source, table, name) for name in info["files"])
        db.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet([{files[1:-1]}]);")

    gt_uuids = sql_list(ground_truth_uuids(dataset_cfg))
    gt_node_ids = sorted(
        row[0]
        for row in db.execute(
            " UNION ".join(
                f"SELECT CAST(index_id AS VARCHAR) FROM {table} WHERE node_uuid IN {gt_uuids}"
                for table in TABLE_COLUMNS
                if table != "event_table"
            )
        ).fetchall()
    )

    columns = ", ".join(TABLE_COLUMNS["event_table"])
    queries = [
        f"(SELECT {columns} FROM event_table WHERE timestamp_rec >= {start} AND "
        f"timestamp_rec < {end} ORDER BY timestamp_rec, event_uuid LIMIT {args.events_per_day})"
        for _, start, end in day_ranges(dataset_cfg)
    ]
    queries += [
        f"(SELECT {columns} FROM event_table WHERE timestamp_rec BETWEEN {start} AND {end} "
        f"AND src_index_id IN {sql_list(gt_node_ids)} AND dst_index_id IN {sql_list(gt_node_ids)})"
        for start, end in attack_windows(dataset_cfg)
    ]
    events = f"SELECT * FROM ({' UNION '.join(queries)}) ORDER BY timestamp_rec, event_uuid"
    db.execute(f"CREATE TABLE sample_events AS {events};")
    node_ids = sql_list(
        sorted(
            {
                str(node_id)
                for row in db.execute(
                    "SELECT src_index_id, dst_index_id FROM sample_events"
                ).fetchall()
                for node_id in row
            }
        )
    )
    queries = {"event_table": "SELECT * FROM sample_events"}
    for table, columns in TABLE_COLUMNS.items():
        if table != "event_table":
            queries[table] = (
                f"SELECT {', '.join(columns)} FROM {table} WHERE CAST(index_id AS VARCHAR) IN "
                f"{node_ids} OR node_uuid IN {gt_uuids} ORDER BY index_id"
            )

    shutil.rmtree(SAMPLE_DIR, ignore_errors=True)
    manifest = {"format_version": 1, "database": DATASET.lower(), "tables": {}}
    for table, query in queries.items():
        os.makedirs(os.path.join(SAMPLE_DIR, table))
        path = os.path.join(SAMPLE_DIR, table, "part-00000.parquet")
        db.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet, COMPRESSION zstd);")
        num_rows = db.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
        manifest["tables"][table] = {"files": ["part-00000.parquet"], "num_rows": num_rows}
        print(f"{table}: {num_rows} rows")
    with open(os.path.join(SAMPLE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    db.close()
    print(f"Sample written to {SAMPLE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets_dir",
        default=None,
        help="Folder of the full dataset (default: $PIDSMAKER_DATASETS_DIR, else "
        "~/.cache/pidsmaker/datasets)",
    )
    parser.add_argument("--events_per_day", type=int, default=5000)
    main(parser.parse_args())
