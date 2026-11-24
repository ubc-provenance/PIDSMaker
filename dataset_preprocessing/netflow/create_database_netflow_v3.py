"""NetFlow v3 preprocessing: creates the PostgreSQL database of a NetFlow v3 dataset.

The NetFlow v3 datasets of the University of Queensland (Luay et al., "Temporal Analysis of NetFlow
Datasets for Network Intrusion Detection Systems", 2025) are network flows with 53 NetFlow
features, including their absolute start time, labelled per flow. E.g. NF-CSE-CIC-IDS2018-v3, the
NetFlow version of CSE-CIC-IDS2018 (used by Anomal-E in its v2 version, which has no timestamps).
Download the dataset from https://staff.itee.uq.edu.au/marius/NIDS_datasets/, then run:

    python dataset_preprocessing/netflow/create_database_netflow_v3.py CIC_IDS2018 \
        --input /data/NF-CSE-CIC-IDS2018-v3.csv

The input is the CSV file of the dataset (also gzipped or zipped), or a parquet version of it
(requires pyarrow).

The database follows the `netflow_v3` schema (see `pidsmaker/config/schemas.py`): nodes are hosts
(IP addresses), events are flows from the source to the destination host, typed by their
transport protocol, whose attributes are the ports (service ports, or IANA range), the bytes and
packets in each direction and the duration, and malicious flows have `label=1`.
"""

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.config.schemas import NETFLOW_PORTS
from pidsmaker.utils.dataset_utils import event_table_columns, get_schema
from pidsmaker.utils.schema_loading import (
    copy_rows,
    create_tables,
    ensure_database,
    get_database_name,
    index_events_by_time,
)
from pidsmaker.utils.utils import init_database_connection, log, ns_time_to_datetime_US, stringtomd5

COLUMNS = [
    "FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
    "Label",
]  # fmt: skip
PROTOCOLS = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}
_SERVICE_PORTS = np.array(sorted(int(p) for p in NETFLOW_PORTS if p.isdigit()))
NUMERIC = {
    "in_bytes": "IN_BYTES",
    "out_bytes": "OUT_BYTES",
    "in_pkts": "IN_PKTS",
    "out_pkts": "OUT_PKTS",
    "duration": "FLOW_DURATION_MILLISECONDS",
}


def port_categories(ports):
    """Categories of ports: service ports are kept, others are grouped by IANA range."""
    ports = pd.to_numeric(ports, errors="coerce")
    out = np.where(
        ports < 1024, "1-1023", np.where(ports < 49152, "1024-49151", "49152+")
    ).astype(object)
    out[np.isin(ports, _SERVICE_PORTS)] = ports[np.isin(ports, _SERVICE_PORTS)].astype(int).astype(str)
    out[ports.isna().to_numpy()] = None
    return out


def read_chunks(path, chunk_size):
    """DataFrames of the `COLUMNS` of the dataset, in file order."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size, columns=COLUMNS):
            yield batch.to_pandas()
        return
    if path.endswith(".zip"):
        archive = zipfile.ZipFile(path)
        name = next(n for n in archive.namelist() if n.endswith(".csv") and "Features" not in n)
        source = archive.open(name)
    else:
        source = path  # pandas decompresses .gz files
    yield from pd.read_csv(source, usecols=COLUMNS, chunksize=chunk_size)


def load_events(cfg, path, chunk_size):
    schema = get_schema(cfg)
    columns = event_table_columns(schema)
    cur, connect = init_database_connection(cfg)
    host2id = {}
    num_events = num_malicious = 0
    first = last = None
    for chunk in read_chunks(path, chunk_size):
        # Hosts are numbered in order of appearance
        for ip in pd.unique(chunk[["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]].to_numpy().ravel()):
            host2id.setdefault(ip, len(host2id))
        src, dst = chunk["IPV4_SRC_ADDR"], chunk["IPV4_DST_ADDR"]
        t = chunk["FLOW_START_MILLISECONDS"].astype(np.int64) * 1_000_000
        rows = pd.DataFrame(
            {
                "src_node": src,
                "src_index_id": src.map(host2id),
                "operation": chunk["PROTOCOL"].map(PROTOCOLS).fillna("OTHER"),
                "dst_node": dst,
                "dst_index_id": dst.map(host2id),
                "event_uuid": np.arange(num_events, num_events + len(chunk)).astype(str),
                "timestamp_rec": t,
                "src_port": port_categories(chunk["L4_SRC_PORT"]),
                "dst_port": port_categories(chunk["L4_DST_PORT"]),
                **{name: chunk[column] for name, column in NUMERIC.items()},
                "label": (chunk["Label"] > 0).astype(int),
            }
        )[columns]
        buffer = io.StringIO()
        rows.to_csv(buffer, header=False, index=False)
        buffer.seek(0)
        cur.copy_expert(
            f"COPY {schema.event_table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)", buffer
        )
        connect.commit()
        num_events += len(chunk)
        num_malicious += int(rows["label"].sum())
        first = t.min() if first is None else min(first, t.min())
        last = t.max() if last is None else max(last, t.max())
        log(f"Loaded {num_events:,} flows ({num_malicious:,} malicious), {len(host2id):,} hosts")

    copy_rows(
        cur,
        schema.node_type("host").table,
        ["node_uuid", "hash_id", "name", "index_id"],
        ((ip, stringtomd5(ip), ip, i) for ip, i in host2id.items()),
    )
    connect.commit()
    connect.close()
    tz = cfg.dataset.timezone
    log(
        f"Flows: {num_events:,} ({num_malicious:,} malicious), hosts: {len(host2id):,}, from "
        f"{ns_time_to_datetime_US(first, tz)} to {ns_time_to_datetime_US(last, tz)} ({tz})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="CIC_IDS2018", help="Dataset name in the config")
    parser.add_argument("--input", required=True, help="CSV (.csv, .csv.gz, .zip) or parquet file")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="Flows loaded at once")
    parser.add_argument("--database", default=None, help="Overrides the database of the dataset")
    parser.add_argument("--overwrite", action="store_true", help="Recreates existing tables")
    args, unknown = parser.parse_known_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    # system config (velox) is irrelevant: only the dataset and database settings are used
    cfg_args, _ = get_runtime_required_args(
        return_unknown_args=True, args=["velox", args.dataset] + unknown
    )
    cfg = get_yml_cfg(cfg_args)
    if args.database:
        cfg.dataset.database = cfg.dataset.database_all_file = args.database

    ensure_database(cfg)
    create_tables(cfg, args.overwrite)
    load_events(cfg, args.input, args.chunk_size)
    index_events_by_time(cfg)
    log(f"Database {get_database_name(cfg)} created.")


if __name__ == "__main__":
    main()
