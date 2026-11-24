"""LANL dataset preprocessing: creates the PostgreSQL database of the LANL events.

Dataset: "Comprehensive, Multi-Source Cyber-Security Events" (A. D. Kent, 2015),
https://csr.lanl.gov/data/cyber1/. Two sources are supported:

- the preprocessed version of LARES (recommended), with NTLM authentications and network flows
  split in files of one minute, labelled as in the LARES paper (see `--lares_dir`; flows are
  loaded with `--with_flows`, as they are ~3 times more numerous than authentications). Download
  `lanl_optc_datasets.tar.gz` from the LARES repository (https://github.com/TristanBilot/lares),
  then run:

      python dataset_preprocessing/lanl/create_database_lanl.py LANL \
          --lares_dir /data/lanl_optc_datasets/LANL/preprocessed

- the raw files `auth.txt.gz` and `redteam.txt.gz` (authentications only, see `--auth_file`),
  requested on the LANL website:

      python dataset_preprocessing/lanl/create_database_lanl.py LANL \
          --auth_file /data/lanl/auth.txt.gz --redteam_file /data/lanl/redteam.txt.gz

The database follows the `lanl_auth` schema (see `pidsmaker/config/schemas.py`):

- nodes are the computers (`host_node_table`), identified by their name (e.g. C17693);
- authentications go from a source to a destination computer. Their `operation` is the
  authentication orientation (LogOn, LogOff, ...), and their attributes are the logon type,
  authentication type, success, and the kinds of source and destination users;
- flows (LARES source) have the `operation` FLOW, and their attributes are the source and
  destination ports, the protocol, the duration and the numbers of packets and bytes;
- malicious events have `label=1`. With the raw files, authentications matching a red team event
  (same time, source user, source and destination computers) are malicious. With the LARES
  source, the labels of LARES are used: authentications with the same time, source and
  destination computers as a red team event, and flows between the computers of a red team event
  within 5 minutes of it.

As in Euler, Argus, Jbeil and LARES, only NTLM authentications are kept (with the raw files,
`--all_auth_types` keeps all of them). Events whose source or destination computer is unknown
(`?`) are skipped. Self-loops are kept: they can be removed with the `remove_self_loops`
cleaning method.

LANL timestamps are seconds from an undisclosed origin (starting at 1). They are mapped to
synthetic UTC dates such that the first red team event (t=150885) starts on 2015-01-03 00:00:00:
the two first days only contain benign events, and can be used for training and validation.
"""

import argparse
import gzip
import os
import pickle
import re
from datetime import datetime, timedelta, timezone

from tqdm import tqdm

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.config.schemas import LANL_AUTH_ORIENTATIONS
from pidsmaker.utils.dataset_utils import event_table_columns, get_schema
from pidsmaker.utils.schema_loading import (
    copy_rows,
    create_tables,
    ensure_database,
    get_database_name,
    index_events_by_time,
)
from pidsmaker.utils.utils import init_database_connection, log, stringtomd5

# The first red team event of LANL (t=150885) is mapped to 2015-01-03 00:00:00 UTC
FIRST_REDTEAM_TIME = 150885
ORIGIN = datetime(2015, 1, 3, tzinfo=timezone.utc) - timedelta(seconds=FIRST_REDTEAM_TIME)
ORIGIN_NS = int(ORIGIN.timestamp()) * 1_000_000_000

COPY_BATCH = 1_000_000

_USER_RE = re.compile(r"^U\d+(@|$)")
_COMPUTER_ACCOUNT_RE = re.compile(r"^C\d+\$?(@|$)")


def lanl_time_to_ns(t):
    """Synthetic timestamp (ns) of a LANL time (seconds)."""
    return ORIGIN_NS + int(t) * 1_000_000_000


def user_kind(user):
    """Kind of a LANL user: `user` (U123@DOM1), `computer` (C123$@DOM1), `anonymous`, or
    `system` (well-known accounts like SYSTEM@C123 or LOCAL SERVICE@C123)."""
    if _USER_RE.match(user):
        return "user"
    if _COMPUTER_ACCOUNT_RE.match(user):
        return "computer"
    if user.upper().startswith("ANONYMOUS"):
        return "anonymous"
    return "system"


def port_category(port):
    """De-identified ports (N123) are all mapped to `N`."""
    return "N" if port.startswith("N") else port


def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


class EventWriter:
    """Buffers event rows (in the column order of the event table) and bulk loads them."""

    def __init__(self, cfg):
        self.schema = get_schema(cfg)
        self.columns = event_table_columns(self.schema)
        self.attributes = [a.name for a in self.schema.edge_attributes]
        self.cur, self.connect = init_database_connection(cfg)
        self.rows = []
        self.host2id = {}
        self.stats = {"events": 0, "malicious": 0}

    def host_id(self, name):
        if name not in self.host2id:
            self.host2id[name] = len(self.host2id)
        return self.host2id[name]

    def add(self, src, dst, operation, event_uuid, t, attrs, label):
        src_id, dst_id = self.host_id(src), self.host_id(dst)
        row = [src, src_id, operation, dst, dst_id, event_uuid, lanl_time_to_ns(t)]
        row += [attrs.get(name) for name in self.attributes]
        row.append(int(label))
        self.rows.append(row)
        self.stats["events"] += 1
        self.stats["malicious"] += int(label)
        if len(self.rows) >= COPY_BATCH:
            self.flush()

    def flush(self):
        if self.rows:
            copy_rows(self.cur, self.schema.event_table, self.columns, self.rows)
            self.connect.commit()
            self.rows = []

    def close(self):
        self.flush()
        self.connect.close()


def auth_attrs(src_user, dst_user, auth_type, logon_type, success):
    return {
        "logon_type": logon_type,
        "auth_type": auth_type,
        "success": success,
        "src_user_kind": user_kind(src_user),
        "dst_user_kind": user_kind(dst_user),
    }


# --------------------------------------------------------------------------------------
# Raw LANL files
# --------------------------------------------------------------------------------------


def read_redteam(path):
    """{(time, user, src computer, dst computer)} of the red team events."""
    events = set()
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                t, user, src, dst = line.split(",")
                events.add((int(t), user, src, dst))
    return events


def load_raw(writer, auth_file, redteam, all_auth_types, max_time):
    orientations = set(LANL_AUTH_ORIENTATIONS)
    skipped = {"not_ntlm": 0, "unknown_host": 0, "malformed": 0}
    found_redteam = set()
    with open_text(auth_file) as f:
        for line_no, line in enumerate(tqdm(f, desc="Reading authentications", unit=" lines")):
            fields = line.rstrip("\n").split(",")
            if len(fields) != 9:
                skipped["malformed"] += 1
                continue
            t, src_user, dst_user, src, dst, auth_type, logon_type, orientation, success = fields
            t = int(t)
            if max_time is not None and t > max_time:
                break
            if not all_auth_types and auth_type != "NTLM":
                skipped["not_ntlm"] += 1
                continue
            if src == "?" or dst == "?":
                skipped["unknown_host"] += 1
                continue
            key = (t, src_user, src, dst)
            label = key in redteam
            if label:
                found_redteam.add(key)
            writer.add(
                src,
                dst,
                orientation if orientation in orientations else "OTHER",
                f"a{line_no:010d}",
                t,
                auth_attrs(src_user, dst_user, auth_type, logon_type, success),
                label,
            )
    log(f"Skipped authentications: {skipped}")
    log(f"Red team events: {len(found_redteam)}/{len(redteam)} matched to kept authentications")
    return [k[0] for k in found_redteam]


# --------------------------------------------------------------------------------------
# LARES preprocessed files
# --------------------------------------------------------------------------------------


class _NoGlobalsUnpickler(pickle.Unpickler):
    """The LARES maps are lists of strings: no global (class, function) may be loaded."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Forbidden global {module}.{name} in a LARES map")


def load_map(path):
    with open(path, "rb") as f:
        return _NoGlobalsUnpickler(f).load()


def numbered_files(directory, prefix):
    """Files `{prefix}_{i}.csv` (optionally gzipped) of a directory, sorted by i."""
    pattern = re.compile(rf"{prefix}_(\d+)\.csv(\.gz)?")
    files = [(int(m.group(1)), f) for f in os.listdir(directory) if (m := pattern.fullmatch(f))]
    return [os.path.join(directory, f) for _, f in sorted(files)]


def load_lares(writer, lares_dir, max_time, with_flows):
    auth_dir, flows_dir = os.path.join(lares_dir, "auth"), os.path.join(lares_dir, "flows")
    nmap = load_map(os.path.join(auth_dir, "nmap.pkl"))
    umap = load_map(os.path.join(auth_dir, "umap.pkl"))
    atmap = load_map(os.path.join(auth_dir, "atmap.pkl"))
    ltmap = load_map(os.path.join(auth_dir, "ltmap.pkl"))
    aomap = load_map(os.path.join(auth_dir, "aomap.pkl"))
    smap = load_map(os.path.join(auth_dir, "smap.pkl"))
    pomap = load_map(os.path.join(flows_dir, "pomap.pkl"))
    prmap = load_map(os.path.join(flows_dir, "prmap.pkl"))
    # Host ids are those of LARES
    for name in nmap:
        writer.host_id(name)

    orientations = set(LANL_AUTH_ORIENTATIONS)
    malicious_times = []
    for path in tqdm(numbered_files(auth_dir, "graph"), desc="Reading authentications"):
        file_id = os.path.basename(path).split(".")[0]
        with open_text(path) as f:
            for line_no, line in enumerate(f):
                ts, src, dst, src_u, dst_u, auth_t, logon_t, auth_o, success, label = line.rstrip("\n").split(",")
                t = int(ts)
                if max_time is not None and t > max_time:
                    break
                orientation = aomap[int(auth_o)]
                writer.add(
                    nmap[int(src)],
                    nmap[int(dst)],
                    orientation if orientation in orientations else "OTHER",
                    f"a{file_id}_{line_no:07d}",
                    t,
                    auth_attrs(
                        umap[int(src_u)], umap[int(dst_u)], atmap[int(auth_t)], ltmap[int(logon_t)], smap[int(success)]
                    ),
                    int(label),
                )
                if label == "1":
                    malicious_times.append(t)

    flow_files = numbered_files(flows_dir, "flows") if with_flows else []
    for path in tqdm(flow_files, desc="Reading flows"):
        file_id = os.path.basename(path).split(".")[0]
        with open_text(path) as f:
            for line_no, line in enumerate(f):
                ts, src, dst, src_p, dst_p, proto, duration, packets, num_bytes, label = line.rstrip("\n").split(",")
                t = int(ts)
                if max_time is not None and t > max_time:
                    break
                attrs = {
                    "src_port": port_category(pomap[int(src_p)]),
                    "dst_port": port_category(pomap[int(dst_p)]),
                    "protocol": prmap[int(proto)],
                    "duration": float(duration),
                    "packets": float(packets),
                    "bytes": float(num_bytes),
                }
                writer.add(nmap[int(src)], nmap[int(dst)], "FLOW", f"f{file_id}_{line_no:07d}", t, attrs, int(label))
                if label == "1":
                    malicious_times.append(t)
    return malicious_times


def load_hosts(cfg, host2id):
    schema = get_schema(cfg)
    cur, connect = init_database_connection(cfg)
    copy_rows(
        cur,
        schema.node_type("host").table,
        ["node_uuid", "hash_id", "name", "index_id"],
        ((name, stringtomd5(name), name, index_id) for name, index_id in host2id.items()),
    )
    connect.commit()
    connect.close()
    log(f"Saved {len(host2id)} hosts")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", default="LANL", help="Dataset name in the config")
    parser.add_argument("--lares_dir", help="LANL/preprocessed folder of the LARES datasets")
    parser.add_argument(
        "--with_flows", action="store_true", help="LARES source only: also load the network flows"
    )
    parser.add_argument("--auth_file", help="Path to the raw auth.txt or auth.txt.gz")
    parser.add_argument("--redteam_file", help="Path to the raw redteam.txt or redteam.txt.gz")
    parser.add_argument(
        "--all_auth_types",
        action="store_true",
        help="Raw files only: keep all authentications instead of the NTLM ones only",
    )
    parser.add_argument(
        "--max_time", type=int, default=None, help="Stops reading after this LANL time (seconds)"
    )
    parser.add_argument("--database", default=None, help="Overrides the database of the dataset")
    parser.add_argument("--overwrite", action="store_true", help="Recreates existing tables")
    args, unknown = parser.parse_known_args()
    if bool(args.lares_dir) == bool(args.auth_file):
        parser.error("Use either --lares_dir or --auth_file (with --redteam_file)")
    if args.auth_file and not args.redteam_file:
        parser.error("--auth_file requires --redteam_file")

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
    writer = EventWriter(cfg)
    try:
        if args.lares_dir:
            malicious_times = load_lares(writer, args.lares_dir, args.max_time, args.with_flows)
        else:
            redteam = read_redteam(args.redteam_file)
            log(f"Loaded {len(redteam)} red team events")
            malicious_times = load_raw(writer, args.auth_file, redteam, args.all_auth_types, args.max_time)
    finally:
        writer.close()
    log(f"Events: {writer.stats}")
    if malicious_times:
        first = ORIGIN + timedelta(seconds=min(malicious_times))
        last = ORIGIN + timedelta(seconds=max(malicious_times))
        log(f"Malicious events span {first} to {last} (UTC)")

    index_events_by_time(cfg)
    load_hosts(cfg, writer.host2id)
    log(f"Database {get_database_name(cfg)} created.")


if __name__ == "__main__":
    main()
