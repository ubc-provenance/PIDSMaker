"""
Load provenance_data/ TSV files into a new PostgreSQL database (PROVENANCE_BENIGN)
using the same DARPA TC schema as CADETS/THEIA/etc., so it works on any machine
that supports TC/OPTC datasets.

  entities.tsv: entity_id  entity_type  entity_text
    - entity_id: proc_N, file_N, sock_N
    - entity_type: PROC, FILE, SOCK
    - entity_text: "exepath [cmdline]" | "/path/to/file" | "lip lport rip rport"
  edges.tsv: src_id  event_type  dst_id  timestamp
    - event_type: DARPA TC format (EVENT_READ, EVENT_WRITE, ...)
    - timestamp: Unix float seconds
"""

import argparse
import os
import sys

import psycopg2
from psycopg2 import extras as ex

DB_PARAMS = dict(host="postgres", port=5432, user="postgres", password="postgres")
DB_NAME = "PROVENANCE_BENIGN"

DDL = """
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid  character varying NOT NULL,
    hash_id    character varying NOT NULL,
    path       character varying,
    cmd        character varying,
    index_id   bigint,
    PRIMARY KEY (node_uuid, hash_id)
);

CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid  character varying NOT NULL,
    hash_id    character varying NOT NULL,
    path       character varying,
    index_id   bigint,
    PRIMARY KEY (node_uuid, hash_id)
);

CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid  character varying NOT NULL,
    hash_id    character varying NOT NULL,
    src_addr   character varying,
    src_port   character varying,
    dst_addr   character varying,
    dst_port   character varying,
    index_id   bigint,
    PRIMARY KEY (node_uuid, hash_id)
);

CREATE SEQUENCE IF NOT EXISTS event_table__id_seq;

CREATE TABLE IF NOT EXISTS event_table (
    src_node      character varying,
    src_index_id  character varying,
    operation     character varying,
    dst_node      character varying,
    dst_index_id  character varying,
    event_uuid    character varying NOT NULL,
    timestamp_rec bigint,
    _id           integer NOT NULL DEFAULT nextval('event_table__id_seq'),
    UNIQUE (_id)
);
"""


def create_database():
    conn = psycopg2.connect(database="postgres", **DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    if cur.fetchone():
        print(f"Database '{DB_NAME}' already exists, dropping and recreating...")
        cur.execute(f'DROP DATABASE "{DB_NAME}";')
    cur.execute(f'CREATE DATABASE "{DB_NAME}";')
    cur.close()
    conn.close()
    print(f"Created database '{DB_NAME}'.")


def create_tables(conn):
    cur = conn.cursor()
    cur.execute(DDL)
    conn.commit()
    cur.close()
    print("Tables created.")


def load_entities(conn, entities_path):
    subject_rows = []
    file_rows = []
    netflow_rows = []

    with open(entities_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2:
                continue
            entity_id = parts[0]
            entity_type = parts[1]
            entity_text = parts[2] if len(parts) > 2 else ""

            num_str = entity_id.rsplit("_", 1)[-1]
            try:
                index_id = int(num_str)
            except ValueError:
                continue

            if entity_type == "PROC":
                space_idx = entity_text.find(" ")
                if space_idx == -1:
                    path = entity_text or None
                    cmd = None
                else:
                    path = entity_text[:space_idx]
                    args = entity_text[space_idx + 1:]
                    # If args already start with path, the full command is already present
                    if args.startswith(path):
                        cmd = args
                    else:
                        cmd = entity_text
                # node_uuid = hash_id = entity_id
                subject_rows.append((entity_id, entity_id, path, cmd, index_id))

            elif entity_type == "FILE":
                file_rows.append((entity_id, entity_id, entity_text or None, index_id))

            elif entity_type == "SOCK":
                tok = entity_text.split()
                src_addr = tok[0] if len(tok) > 0 else None
                src_port = tok[1] if len(tok) > 1 else None
                dst_addr = tok[2] if len(tok) > 2 else None
                dst_port = tok[3] if len(tok) > 3 else None
                netflow_rows.append((entity_id, entity_id, src_addr, src_port, dst_addr, dst_port, index_id))

    cur = conn.cursor()
    ex.execute_values(
        cur,
        "INSERT INTO subject_node_table (node_uuid, hash_id, path, cmd, index_id) VALUES %s ON CONFLICT DO NOTHING",
        subject_rows, page_size=5000,
    )
    ex.execute_values(
        cur,
        "INSERT INTO file_node_table (node_uuid, hash_id, path, index_id) VALUES %s ON CONFLICT DO NOTHING",
        file_rows, page_size=5000,
    )
    ex.execute_values(
        cur,
        "INSERT INTO netflow_node_table (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id) VALUES %s ON CONFLICT DO NOTHING",
        netflow_rows, page_size=5000,
    )
    conn.commit()
    cur.close()
    print(f"Loaded {len(subject_rows)} subjects, {len(file_rows)} files, {len(netflow_rows)} netflows.")


def load_edges(conn, edges_path, entity_index_map):
    rows = []
    skipped = 0
    for i, line in enumerate(open(edges_path, "r")):
        if i == 0:
            continue  # skip header
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        src_id, evt_type, dst_id, ts_str = parts[0], parts[1], parts[2], parts[3]
        try:
            ts_ns = int(float(ts_str) * 1e9)
        except ValueError:
            skipped += 1
            continue

        src_idx = entity_index_map.get(src_id)
        dst_idx = entity_index_map.get(dst_id)
        if src_idx is None or dst_idx is None:
            skipped += 1
            continue

        event_uuid = str(i)
        rows.append((src_id, str(src_idx), evt_type, dst_id, str(dst_idx), event_uuid, ts_ns))

    cur = conn.cursor()
    ex.execute_values(
        cur,
        """INSERT INTO event_table
           (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec)
           VALUES %s""",
        rows, page_size=10000,
    )
    conn.commit()
    cur.close()
    print(f"Loaded {len(rows)} edges ({skipped} skipped).")


def build_entity_index_map(entities_path):
    """Returns entity_id -> index_id (int)."""
    mapping = {}
    with open(entities_path, "r") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 1:
                continue
            entity_id = parts[0]
            num_str = entity_id.rsplit("_", 1)[-1]
            try:
                mapping[entity_id] = int(num_str)
            except ValueError:
                continue
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Load provenance_data TSV into PROVENANCE_BENIGN DB (DARPA TC schema)")
    parser.add_argument("--data-dir", default="/data/provenance_data",
                        help="Path to folder containing entities.tsv and edges.tsv")
    parser.add_argument("--db-host", default="postgres", help="Postgres host (default: postgres)")
    parser.add_argument("--db-port", type=int, default=5432, help="Postgres port (default: 5432)")
    args = parser.parse_args()

    DB_PARAMS["host"] = args.db_host
    DB_PARAMS["port"] = args.db_port

    entities_path = os.path.join(args.data_dir, "entities.tsv")
    edges_path = os.path.join(args.data_dir, "edges.tsv")

    for p in (entities_path, edges_path):
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            sys.exit(1)

    create_database()

    conn = psycopg2.connect(database=DB_NAME, **DB_PARAMS)
    try:
        create_tables(conn)
        entity_index_map = build_entity_index_map(entities_path)
        load_entities(conn, entities_path)
        load_edges(conn, edges_path, entity_index_map)
    finally:
        conn.close()

    print(f"\nDone. Database '{DB_NAME}' is ready.")
    print(f"To dump: pg_dump -U postgres -h localhost -p 5432 -F c -d {DB_NAME} -f /data/{DB_NAME}.dump")


if __name__ == "__main__":
    main()
