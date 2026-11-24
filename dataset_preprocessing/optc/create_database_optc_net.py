"""DARPA OpTC network preprocessing: creates the PostgreSQL database of the network flows of OpTC.

The network version of OpTC is made of the FLOW START and FLOW MESSAGE events of the eCAR data of
all the hosts of the exercise. Both are loaded in the database; as in LARES and Argus, only FLOW
START events are used by default (`construction.operations` of the `OPTC_NET` dataset). This script loads the version
preprocessed by LARES (https://github.com/TristanBilot/lares), split in files of one minute and
labelled as in the LARES paper. Download `lanl_optc_datasets.tar.gz` from the LARES repository,
then run:

    python dataset_preprocessing/optc/create_database_optc_net.py OPTC_NET \
        --lares_dir /data/lanl_optc_datasets/OPTC/preprocessed

Each line of the LARES files is `timestamp,action,label,src,dst,src_port,dst_port,protocol,size,idx`
where `action` is F_S (FLOW START) or F_M (FLOW MESSAGE), hosts are integer ids and `size` is the
size of messages (empty for FLOW START events).

The database follows the `optc_flows` schema (see `pidsmaker/config/schemas.py`): nodes are hosts,
events are flows from a source to a destination host whose attributes are the ports (grouped as
in LARES), the protocol and the size of messages, and malicious events have `label=1`.
Timestamps are the real timestamps of OpTC (seconds since epoch).

The official eCAR files of OpTC can be downloaded with `download_optc_net.py`.
"""

import argparse
import gzip
import io
import multiprocessing
import os
import re

import psycopg2
from tqdm import tqdm

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.config.schemas import OPTC_PORTS
from pidsmaker.utils.dataset_utils import event_table_columns, get_schema
from pidsmaker.utils.schema_loading import (
    copy_rows,
    create_tables,
    ensure_database,
    get_database_name,
    index_events_by_time,
)
from pidsmaker.utils.utils import init_database_connection, log, ns_time_to_datetime_US, stringtomd5

ACTIONS = {"F_S": "FLOW_START", "F_M": "FLOW_MESSAGE"}
_SPECIFIC_PORTS = {p for p in OPTC_PORTS if p.isdigit()}


def port_category(port):
    """Ports grouped as in LARES: common service ports are kept, others are grouped by range."""
    if port == "":
        return None
    port = int(float(port))
    if str(port) in _SPECIFIC_PORTS:
        return str(port)
    if port < 1024:
        return "1-1023"
    if port < 5355:
        return "1024-5354"
    if port <= 50_000:
        return "5356-50000"
    if port <= 60_000:
        return "50001-60000"
    return "60001+"


def numbered_files(directory):
    """Files `{i}.csv.gz` of a directory, sorted by i (other files, e.g. macOS `._*`, are ignored)."""
    pattern = re.compile(r"(\d+)\.csv(\.gz)?")
    files = [(int(m.group(1)), f) for f in os.listdir(directory) if (m := pattern.fullmatch(f))]
    return [os.path.join(directory, f) for _, f in sorted(files)]


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


_worker = {}


def _init_worker(db_params, table, columns, attributes):
    _worker["connect"] = psycopg2.connect(**db_params)
    _worker.update(table=table, columns=columns, attributes=attributes)


def _csv(value):
    return "" if value is None else str(value)


def _load_file(path):
    """Loads the events of one LARES file with COPY (each worker has its own connection, so
    files are loaded in parallel), and returns its hosts and statistics."""
    file_id = os.path.basename(path).split(".")[0]
    attributes = _worker["attributes"]
    lines, hosts = [], set()
    stats = {"events": 0, "malicious": 0, "unknown_action": 0}
    min_mal, max_mal = None, None
    with open_text(path) as f:
        for line_no, line in enumerate(f):
            ts, action, label, src, dst, src_port, dst_port, protocol, size, _idx = line.rstrip("\n").split(",")
            operation = ACTIONS.get(action)
            if operation is None:
                stats["unknown_action"] += 1
                operation = "OTHER"
            attrs = {
                "src_port": port_category(src_port),
                "dst_port": port_category(dst_port),
                "protocol": protocol or None,
                "size": float(size) if size else None,
            }
            t = int(round(float(ts) * 1e9))
            label = int(label or 0)
            fields = [src, src, operation, dst, dst, f"{file_id}_{line_no:07d}", t]
            fields += [attrs[name] for name in attributes] + [label]
            lines.append(",".join(map(_csv, fields)))
            hosts.update((src, dst))
            stats["events"] += 1
            if label:
                stats["malicious"] += 1
                min_mal = t if min_mal is None else min(min_mal, t)
                max_mal = t if max_mal is None else max(max_mal, t)
    if lines:
        connect = _worker["connect"]
        with connect.cursor() as cur:
            cur.copy_expert(
                f"COPY {_worker['table']} ({', '.join(_worker['columns'])}) FROM STDIN WITH (FORMAT csv)",
                io.StringIO("\n".join(lines) + "\n"),
            )
        connect.commit()
    return hosts, stats, min_mal, max_mal


def load_events(cfg, lares_dir, num_workers):
    schema = get_schema(cfg)
    columns = event_table_columns(schema)
    attributes = [a.name for a in schema.edge_attributes]
    db_params = dict(
        dbname=get_database_name(cfg),
        host=cfg.database.host,
        user=cfg.database.user,
        password=cfg.database.password,
        port=cfg.database.port,
    )
    hosts = set()
    stats = {"events": 0, "malicious": 0, "unknown_action": 0}
    malicious_times = []
    files = numbered_files(lares_dir)
    init_args = (db_params, schema.event_table, columns, attributes)
    with multiprocessing.Pool(num_workers, initializer=_init_worker, initargs=init_args) as pool:
        for file_hosts, file_stats, min_mal, max_mal in tqdm(
            pool.imap_unordered(_load_file, files), total=len(files), desc="Loading flows"
        ):
            hosts |= file_hosts
            for k, v in file_stats.items():
                stats[k] += v
            malicious_times += [t for t in (min_mal, max_mal) if t is not None]

    # Hosts keep their LARES integer id as index id
    cur, connect = init_database_connection(cfg)
    copy_rows(
        cur,
        schema.node_type("host").table,
        ["node_uuid", "hash_id", "name", "index_id"],
        ((h, stringtomd5(h), h, int(h)) for h in sorted(hosts, key=int)),
    )
    connect.commit()
    connect.close()

    log(f"Events: {stats}, hosts: {len(hosts)}")
    if malicious_times:
        tz = cfg.dataset.timezone
        log(
            f"Malicious events span {ns_time_to_datetime_US(min(malicious_times), tz)} to "
            f"{ns_time_to_datetime_US(max(malicious_times), tz)} ({tz})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="OPTC_NET", help="Dataset name in the config")
    parser.add_argument("--lares_dir", required=True, help="OPTC/preprocessed folder of the LARES datasets")
    parser.add_argument("--num_workers", type=int, default=16, help="Processes reading the files")
    parser.add_argument("--database", default=None, help="Overrides the database of the dataset")
    parser.add_argument("--overwrite", action="store_true", help="Recreates existing tables")
    args, unknown = parser.parse_known_args()

    # Remaining args (e.g. --database_host) are handled by the framework's parser. The
    # system config (velox) is irrelevant: only the dataset and database settings are used
    cfg_args, _ = get_runtime_required_args(
        return_unknown_args=True, args=["velox", args.dataset] + unknown
    )
    cfg = get_yml_cfg(cfg_args)
    if args.database:
        cfg.dataset.database = cfg.dataset.database_all_file = args.database

    ensure_database(cfg)
    create_tables(cfg, args.overwrite)
    load_events(cfg, args.lares_dir, args.num_workers)
    index_events_by_time(cfg)
    log(f"Database {get_database_name(cfg)} created.")


if __name__ == "__main__":
    main()
