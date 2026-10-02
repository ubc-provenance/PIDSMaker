from pidsmaker.config import get_node_feats_from_cfg
from pidsmaker.utils.utils import init_database_connection

import torch
import os

# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_EXECUTE",
    "EVENT_LSEEK",
    "EVENT_MMAP",
    "EVENT_OPEN",
    "EVENT_ACCEPT",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES",
    "READ",
]

# The following edges are not considered to construct the
# temporal graph for experiments.
exclude_edge_type = set(
    [
        "EVENT_FCNTL",  # EVENT_FCNTL does not have any predicate
        "EVENT_OTHER",  # EVENT_OTHER does not have any predicate
        "EVENT_ADD_OBJECT_ATTRIBUTE",  # This is used to add attributes to an object that was incomplete at the time of publish
        "EVENT_FLOWS_TO",  # No corresponding system call event
    ]
)

rel2id_darpa_tc = {
    1: "EVENT_CONNECT",
    "EVENT_CONNECT": 1,
    2: "EVENT_EXECUTE",
    "EVENT_EXECUTE": 2,
    3: "EVENT_OPEN",
    "EVENT_OPEN": 3,
    4: "EVENT_READ",
    "EVENT_READ": 4,
    5: "EVENT_RECVFROM",
    "EVENT_RECVFROM": 5,
    6: "EVENT_RECVMSG",
    "EVENT_RECVMSG": 6,
    7: "EVENT_SENDMSG",
    "EVENT_SENDMSG": 7,
    8: "EVENT_SENDTO",
    "EVENT_SENDTO": 8,
    9: "EVENT_WRITE",
    "EVENT_WRITE": 9,
    10: "EVENT_CLONE",
    "EVENT_CLONE": 10,
}

possible_events = {
    ("subject", "subject"): [
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_OPEN",
        "EVENT_CONNECT",
        "EVENT_RECVFROM",
        "EVENT_SENDTO",
        "EVENT_CLONE",
        "EVENT_SENDMSG",
        "EVENT_RECVMSG",
    ],
    ("subject", "file"): [
        "EVENT_WRITE",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
        "EVENT_SENDTO",
        "EVENT_CLONE",
    ],
    ("subject", "netflow"): [
        "EVENT_WRITE",
        "EVENT_SENDTO",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
    ],
    ("file", "subject"): [
        "EVENT_READ",
        "EVENT_OPEN",
        "EVENT_RECVFROM",
        "EVENT_EXECUTE",
        "EVENT_RECVMSG",
    ],
    ("netflow", "subject"): [
        "EVENT_OPEN",
        "EVENT_READ",
        "EVENT_RECVFROM",
        "EVENT_RECVMSG",
    ],
}
rel2id_optc = {
    1: "OPEN",
    "OPEN": 1,
    2: "READ",
    "READ": 2,
    3: "CREATE",
    "CREATE": 3,
    4: "MESSAGE",
    "MESSAGE": 4,
    5: "MODIFY",
    "MODIFY": 5,
    6: "START",
    "START": 6,
    7: "RENAME",
    "RENAME": 7,
    8: "DELETE",
    "DELETE": 8,
    9: "TERMINATE",
    "TERMINATE": 9,
    10: "WRITE",
    "WRITE": 10,
}

# Vocabulary for OpTC graphs built with consistent_edge_types=True.
# Translated edges use the same IDs as rel2id_darpa_tc (enabling shared weights).
# Untranslated OpTC-only edges (CREATE/DELETE/RENAME for subject→file,
# TERMINATE for subject→subject) get new IDs starting at 11.
rel2id_optc_consistent = {
    # Shared TC types — same IDs as rel2id_darpa_tc
    1: "EVENT_CONNECT",   "EVENT_CONNECT": 1,
    3: "EVENT_OPEN",      "EVENT_OPEN": 3,
    4: "EVENT_READ",      "EVENT_READ": 4,
    5: "EVENT_RECVFROM",  "EVENT_RECVFROM": 5,
    6: "EVENT_RECVMSG",   "EVENT_RECVMSG": 6,
    7: "EVENT_SENDMSG",   "EVENT_SENDMSG": 7,
    9: "EVENT_WRITE",     "EVENT_WRITE": 9,
    10: "EVENT_CLONE",    "EVENT_CLONE": 10,
    # OpTC-specific types that are not translated
    11: "CREATE",     "CREATE": 11,
    12: "DELETE",     "DELETE": 12,
    13: "RENAME",     "RENAME": 13,
    14: "TERMINATE",  "TERMINATE": 14,
}
possible_events_optc = {
    ("subject", "subject"): [
        "CREATE",
        "OPEN",
        "TERMINATE",
    ],
    ("subject", "file"): [
        "CREATE",
        "DELETE",
        "MODIFY",
        "RENAME",
        "WRITE",
    ],
    ("subject", "netflow"): [
        "MESSAGE",
        "START",
    ],
    ("file", "subject"): [
        "READ",
    ],
    ("netflow", "subject"): [
        "MESSAGE",
        "OPEN",
        "START",
    ],
}

# Context-dependent mapping from OpTC edge types to DARPA TC edge types.
# Key: (src_type, optc_label, dst_type) -> tc_label
# This enables shared vocabulary between OpTC and TC for transfer learning.
OPTC_TO_TC_EDGE_MAP = {
    # subject -> subject (process lifecycle)
    ("subject", "CREATE", "subject"): "EVENT_CLONE",
    ("subject", "START", "subject"): "EVENT_CLONE",
    ("subject", "OPEN", "subject"): "EVENT_OPEN",
    # subject -> file (file operations)
    ("subject", "MODIFY", "file"): "EVENT_WRITE",
    ("subject", "WRITE", "file"): "EVENT_WRITE",
    # subject -> netflow (outbound network)
    ("subject", "MESSAGE", "netflow"): "EVENT_SENDMSG",
    ("subject", "START", "netflow"): "EVENT_CONNECT",
    # file -> subject (reversed reads)
    ("file", "READ", "subject"): "EVENT_READ",
    # netflow -> subject (reversed inbound network)
    ("netflow", "MESSAGE", "subject"): "EVENT_RECVMSG",
    ("netflow", "OPEN", "subject"): "EVENT_OPEN",
    ("netflow", "START", "subject"): "EVENT_RECVFROM",
}

# possible_events for OpTC when using consistent (TC) edge types
possible_events_optc_consistent = {
    ("subject", "subject"): [
        "EVENT_CLONE",
        "EVENT_OPEN",
        "TERMINATE",
    ],
    ("subject", "file"): [
        "CREATE",
        "DELETE",
        "RENAME",
        "EVENT_WRITE",
    ],
    ("subject", "netflow"): [
        "EVENT_SENDMSG",
        "EVENT_CONNECT",
    ],
    ("file", "subject"): [
        "EVENT_READ",
    ],
    ("netflow", "subject"): [
        "EVENT_RECVMSG",
        "EVENT_OPEN",
        "EVENT_RECVFROM",
    ],
}


def translate_optc_to_tc(label, src_type, dst_type):
    """Translate an OpTC edge label to its DARPA TC equivalent.

    Uses (src_type, label, dst_type) context to disambiguate edge types
    that map differently depending on direction (e.g. MESSAGE -> SENDMSG/RECVMSG).

    Returns the TC label, or the original label if no mapping exists.
    """
    return OPTC_TO_TC_EDGE_MAP.get((src_type, label, dst_type), label)

rel2id_atlasv2 = {
    0: "ACTION_FILE_UNDELETE",
    1: "ACTION_FILE_OPEN_SET_ATTRIBUTES",
    2: "ACTION_FILE_CREATE",
    3: "ACTION_FILE_OPEN_DELETE",
    4: "ACTION_FILE_OPEN_SET_SECURITY",
    5: "ACTION_FILE_TRUNCATE",
    6: "ACTION_FILE_MOD_OPEN",
    7: "ACTION_FILE_DELETE",
    8: "ACTION_FILE_LAST_WRITE",
    9: "ACTION_FILE_OPEN_WRITE",
    10: "ACTION_FILE_RENAME",
    11: "ACTION_FILE_OPEN_READ",
    12: "ACTION_FILE_WRITE",
    13: "ACTION_OPEN_KEY_DELETE",
    14: "ACTION_WRITE_VALUE",
    15: "ACTION_DELETE_VALUE",
    16: "ACTION_OPEN_KEY_READ",
    17: "ACTION_DELETE_KEY",
    18: "ACTION_LOAD_KEY",
    19: "ACTION_CREATE_KEY",
    20: "ACTION_OPEN_KEY_WRITE",
    21: "ACTION_LOAD_MODULE",
    22: "ACTION_PROCESS_TERMINATE",
    23: "ACTION_PROCESS_DISCOVERED",
    24: "ACTION_CREATE_PROCESS",
    25: "ACTION_CREATE_PROCESS_EFFECTIVE",
    26: "ACTION_DUP_THREAD_HANDLE",
    27: "ACTION_DUP_PROCESS_HANDLE",
    28: "ACTION_OPEN_PROCESS_HANDLE",
    29: "ACTION_OPEN_THREAD_HANDLE",
    30: "ACTION_LOAD_SCRIPT",
    31: "ACTION_CONNECTION_ESTABLISHED",
    32: "ACTION_CONNECTION_LISTEN",
    33: "ACTION_CONNECTION_CREATE",
    "ACTION_FILE_UNDELETE": 0,
    "ACTION_FILE_OPEN_SET_ATTRIBUTES": 1,
    "ACTION_FILE_CREATE": 2,
    "ACTION_FILE_OPEN_DELETE": 3,
    "ACTION_FILE_OPEN_SET_SECURITY": 4,
    "ACTION_FILE_TRUNCATE": 5,
    "ACTION_FILE_MOD_OPEN": 6,
    "ACTION_FILE_DELETE": 7,
    "ACTION_FILE_LAST_WRITE": 8,
    "ACTION_FILE_OPEN_WRITE": 9,
    "ACTION_FILE_RENAME": 10,
    "ACTION_FILE_OPEN_READ": 11,
    "ACTION_FILE_WRITE": 12,
    "ACTION_OPEN_KEY_DELETE": 13,
    "ACTION_WRITE_VALUE": 14,
    "ACTION_DELETE_VALUE": 15,
    "ACTION_OPEN_KEY_READ": 16,
    "ACTION_DELETE_KEY": 17,
    "ACTION_LOAD_KEY": 18,
    "ACTION_CREATE_KEY": 19,
    "ACTION_OPEN_KEY_WRITE": 20,
    "ACTION_LOAD_MODULE": 21,
    "ACTION_PROCESS_TERMINATE": 22,
    "ACTION_PROCESS_DISCOVERED": 23,
    "ACTION_CREATE_PROCESS": 24,
    "ACTION_CREATE_PROCESS_EFFECTIVE": 25,
    "ACTION_DUP_THREAD_HANDLE": 26,
    "ACTION_DUP_PROCESS_HANDLE": 27,
    "ACTION_OPEN_PROCESS_HANDLE": 28,
    "ACTION_OPEN_THREAD_HANDLE": 29,
    "ACTION_LOAD_SCRIPT": 30,
    "ACTION_CONNECTION_ESTABLISHED": 31,
    "ACTION_CONNECTION_LISTEN": 32,
    "ACTION_CONNECTION_CREATE": 33,
}

rel2id_graph_processor_carbon_black_edr = {
    1: "FILE_OPENED",
    2: "FILE_CLOSED",
    3: "FILE_CREATED",
    4: "FILE_READ",
    5: "FILE_WRITTEN",
    6: "FILE_COPIED",
    7: "FILE_LINKED",
    8: "FILE_RENAMED",
    9: "FILE_TRUNCATED",
    10: "FILE_DELETED",
    11: "FILE_RESTORED",
    12: "NETFLOW_CREATED",
    13: "NETFLOW_CONNECTED",
    14: "NETFLOW_PACKET_RECEIVED",
    15: "NETFLOW_PACKET_SENT",
    16: "NETFLOW_DISCONNECTED",
    17: "PROCESS_EXECUTED",
    18: "PROCESS_KILLED",
    19: "MODULE_LOADED",
    20: "CROSS_PROCESS_INTERACTION",
    21: "REGISTRY_KEY_LOADED",
    22: "REGISTRY_KEY_UNLOADED",
    23: "REGISTRY_KEY_OPENED",
    24: "REGISTRY_KEY_CLOSED",
    25: "REGISTRY_KEY_CREATED",
    26: "REGISTRY_KEY_RENAMED",
    27: "REGISTRY_KEY_REPLACED",
    28: "REGISTRY_KEY_DELETED",
    29: "REGISTRY_KEY_RESTORED",
    30: "REGISTRY_VALUE_CREATED",
    31: "REGISTRY_VALUE_READ",
    32: "REGISTRY_VALUE_WRITTEN",
    33: "REGISTRY_VALUE_DELETED",
    "FILE_OPENED": 1,
    "FILE_CLOSED": 2,
    "FILE_CREATED": 3,
    "FILE_READ": 4,
    "FILE_WRITTEN": 5,
    "FILE_COPIED": 6,
    "FILE_LINKED": 7,
    "FILE_RENAMED": 8,
    "FILE_TRUNCATED": 9,
    "FILE_DELETED": 10,
    "FILE_RESTORED": 11,
    "NETFLOW_CREATED": 12,
    "NETFLOW_CONNECTED": 13,
    "NETFLOW_PACKET_RECEIVED": 14,
    "NETFLOW_PACKET_SENT": 15,
    "NETFLOW_DISCONNECTED": 16,
    "PROCESS_EXECUTED": 17,
    "PROCESS_KILLED": 18,
    "MODULE_LOADED": 19,
    "CROSS_PROCESS_INTERACTION": 20,
    "REGISTRY_KEY_LOADED": 21,
    "REGISTRY_KEY_UNLOADED": 22,
    "REGISTRY_KEY_OPENED": 23,
    "REGISTRY_KEY_CLOSED": 24,
    "REGISTRY_KEY_CREATED": 25,
    "REGISTRY_KEY_RENAMED": 26,
    "REGISTRY_KEY_REPLACED": 27,
    "REGISTRY_KEY_DELETED": 28,
    "REGISTRY_KEY_RESTORED": 29,
    "REGISTRY_VALUE_CREATED": 30,
    "REGISTRY_VALUE_READ": 31,
    "REGISTRY_VALUE_WRITTEN": 32,
    "REGISTRY_VALUE_DELETED": 33,
}

possible_events_darpa_tc = {
    ("subject", "subject"): [
        "EVENT_READ",
        "EVENT_WRITE",
        "EVENT_OPEN",
        "EVENT_CONNECT",
        "EVENT_RECVFROM",
        "EVENT_SENDTO",
        "EVENT_CLONE",
        "EVENT_SENDMSG",
        "EVENT_RECVMSG",
    ],
    ("subject", "file"): [
        "EVENT_WRITE",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
        "EVENT_SENDTO",
        "EVENT_CLONE",
    ],
    ("subject", "netflow"): [
        "EVENT_WRITE",
        "EVENT_SENDTO",
        "EVENT_CONNECT",
        "EVENT_SENDMSG",
    ],
    ("file", "subject"): [
        "EVENT_READ",
        "EVENT_OPEN",
        "EVENT_RECVFROM",
        "EVENT_EXECUTE",
        "EVENT_RECVMSG",
    ],
    ("netflow", "subject"): [
        "EVENT_OPEN",
        "EVENT_READ",
        "EVENT_RECVFROM",
        "EVENT_RECVMSG",
    ],
}

def _has_consistent_edge_types(cfg):
    """Check if consistent_edge_types is enabled in the build_graphs config."""
    try:
        return bool(cfg.construction.consistent_edge_types)
    except (AttributeError, KeyError):
        return False


possible_events_graph_processor_carbon_black_edr = {
    ("subject", "subject"): [
        "PROCESS_EXECUTED",
        "PROCESS_KILLED",
        "CROSS_PROCESS_INTERACTION",
    ],
    ("subject", "file"): [
        "FILE_OPENED",
        "FILE_CLOSED",
        "FILE_CREATED",
        "FILE_WRITTEN",
        "FILE_COPIED",
        "FILE_LINKED",
        "FILE_RENAMED",
        "FILE_TRUNCATED",
        "FILE_DELETED",
        "FILE_RESTORED",
        "REGISTRY_KEY_UNLOADED",
        "REGISTRY_KEY_OPENED",
        "REGISTRY_KEY_CLOSED",
        "REGISTRY_KEY_CREATED",
        "REGISTRY_KEY_RENAMED",
        "REGISTRY_KEY_REPLACED",
        "REGISTRY_KEY_DELETED",
        "REGISTRY_KEY_RESTORED",
        "REGISTRY_VALUE_CREATED",
        "REGISTRY_VALUE_WRITTEN",
        "REGISTRY_VALUE_DELETED",
    ],
    ("subject", "netflow"): [
        "NETFLOW_CREATED",
        "NETFLOW_CONNECTED",
        "NETFLOW_PACKET_SENT",
        "NETFLOW_DISCONNECTED",
    ],
    ("file", "subject"): [
        "FILE_READ",
        "MODULE_LOADED",
        "REGISTRY_KEY_LOADED",
        "REGISTRY_VALUE_READ",
    ],
    ("netflow", "subject"): [
        "NETFLOW_PACKET_RECEIVED",
    ],
}


ntype2id = {
    1: "subject",
    "subject": 1,
    2: "file",
    "file": 2,
    3: "netflow",
    "netflow": 3,
}

DARPA_TC_DATASETS = {
    "CADETS_E3",
    "CADETS_E5",
    "THEIA_E5",
    "THEIA_E3",
    "CLEARSCOPE_E5",
    "CLEARSCOPE_E3",
    "TRACE_E5",
    "TRACE_E3",
    "FIVEDIRECTIONS_E5",
    "FIVEDIRECTIONS_E3",
}
OPTC_DATASETS = {"optc_h201", "optc_h501", "optc_h051"}
ATLASv2_DATASETS = {"atlasv2_h1"}
graph_processor_carbon_black_edr_datasets = {"atlasv2_edr", "carbanakv2_edr"}

OPTC_hostname_map = {
    "optc_h051": "SysClient0051",
    "optc_h201": "SysClient0201",
    "optc_h501": "SysClient0501",
}

DATABASE_FORMAT_MAP = {
    "darpa_tc": {
        "subject": {
            "table_name": "subject_node_table",
            "feature_column_map": {
                "path": "path",
                "cmd_line": "cmd",
            }
        },
        "file": {
            "table_name": "file_node_table",
            "feature_column_map": {
                "path": "path",
            }
        },
        "netflow": {
            "table_name": "netflow_node_table",
            "feature_column_map": {
                "local_ip": "src_addr",
                "local_port": "src_port",
                "remote_ip": "dst_addr",
                "remote_port": "dst_port",
            }
        },
    },
}

def decrement_dict(d):
    return {
        k - 1 if isinstance(k, int) else k: v - 1 if isinstance(v, int) else v for k, v in d.items()
    }


def get_rel2id(cfg, from_zero=False):
    dataset_name = cfg.dataset.name.lower()

    if dataset_name in OPTC_DATASETS and _has_consistent_edge_types(cfg):
        # Graphs were built with a mixed vocabulary: TC-equivalent types (same IDs
        # as rel2id_darpa_tc) + untranslated OpTC-only types (CREATE/DELETE/RENAME/TERMINATE)
        return decrement_dict(rel2id_optc_consistent) if from_zero else rel2id_optc_consistent
    elif dataset_name in OPTC_DATASETS:
        return decrement_dict(rel2id_optc) if from_zero else rel2id_optc
    elif dataset_name in ATLASv2_DATASETS:
        return rel2id_atlasv2
    elif cfg.dataset.name in DARPA_TC_DATASETS:
        return decrement_dict(rel2id_darpa_tc) if from_zero else rel2id_darpa_tc
    elif dataset_name in graph_processor_carbon_black_edr_datasets:
        return decrement_dict(rel2id_graph_processor_carbon_black_edr) if from_zero else rel2id_graph_processor_carbon_black_edr
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_node_map(from_zero=False):
    if from_zero:
        return decrement_dict(ntype2id)
    return ntype2id


def get_num_edge_type(cfg):
    dataset_name = cfg.dataset.name.lower()

    if dataset_name not in OPTC_DATASETS and "edge_type_triplet" in cfg.batching.edge_features:
        possible_events = get_possible_events(cfg)
        return sum([len(events) for events in possible_events.values()])
    if dataset_name in OPTC_DATASETS and _has_consistent_edge_types(cfg):
        max_id = max(v for v in rel2id_optc_consistent.values() if isinstance(v, int))
        return max_id
    return cfg.dataset.num_edge_types


def get_rel2id_considering_triplets(cfg):
    if "edge_type_triplet" in cfg.batching.edge_features:
        possible_events = get_possible_events(cfg)
        return {
            i + 1: e
            for i, e in enumerate(
                [event for events in possible_events.values() for event in events]
            )
        }
    return get_rel2id(cfg)


def get_possible_events(cfg):
    dataset_name = cfg.dataset.name.lower()

    if dataset_name in graph_processor_carbon_black_edr_datasets:
        return possible_events_graph_processor_carbon_black_edr
    else:
        return possible_events_darpa_tc

def uses_darpa_tc_schema(dataset):
    # OPTC and the Carbon Black EDR datasets (atlasv2_edr,
    # carbanakv2_edr) are ingested into the same node/event-table schema as DARPA TC.
    return (
        dataset in DARPA_TC_DATASETS
        or dataset in OPTC_DATASETS
        or dataset.lower() in graph_processor_carbon_black_edr_datasets
    )


def get_database_format(dataset, node_type):
    if uses_darpa_tc_schema(dataset):
        return DATABASE_FORMAT_MAP['darpa_tc'][node_type]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_uuid_to_index_id(cfg):
    dataset = cfg.dataset.name
    node_types = get_node_feats_from_cfg(cfg).keys()

    uuid2nid = {}
    nid2uuid ={}

    cur, conn = init_database_connection(cfg)
    for node_type in node_types:
        table_name = get_database_format(dataset, node_type)['table_name']
        sql = f"SELECT index_id, node_uuid FROM {table_name};"
        cur.execute(sql)
        rows = cur.fetchall()
        for row in rows:
            uuid2nid[str(row[1])] = int(row[0])
            nid2uuid[int(row[0])] = str(row[1])

    return uuid2nid, nid2uuid

def get_time_and_end_node(cfg, start_time, end_time):
    """Return events in the window as (src_index_id, dst_index_id, timestamp, operation).

    src/dst are integer graph index_ids, so callers can compare them directly
    against the index_id-based ground truth.
    """
    dataset = cfg.dataset.name
    cur, conn = init_database_connection(cfg)

    if uses_darpa_tc_schema(dataset):
        sql = f"SELECT src_index_id, dst_index_id, timestamp_rec, operation FROM event_table WHERE timestamp_rec > {start_time} AND timestamp_rec < {end_time};"
        cur.execute(sql)
        data = cur.fetchall()
        return [(int(r[0]), int(r[1]), r[2], r[3]) for r in data]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_node_to_path_and_type(cfg):
    out_path = cfg.construction._node_id_to_path
    out_file = os.path.join(out_path, "node_to_paths.pkl")
    dataset = cfg.dataset.name

    if not os.path.exists(out_file):
        os.makedirs(out_path, exist_ok=True)
        cur, connect = init_database_connection(cfg)

        if uses_darpa_tc_schema(dataset):
            queries = {
                "file": "SELECT index_id, path FROM file_node_table;",
                "netflow": "SELECT index_id, src_addr, dst_addr, src_port, dst_port FROM netflow_node_table;",
                "subject": "SELECT index_id, path, cmd FROM subject_node_table;",
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        node_to_path_type = {}
        for node_type, query in queries.items():
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                if node_type == "netflow":
                    index_id, src_addr, dst_addr, src_port, dst_port = row
                    node_to_path_type[index_id] = {
                        "path": f"{str(src_addr)}:{str(src_port)}->{str(dst_addr)}:{str(dst_port)}",
                        "type": node_type,
                    }
                elif node_type == "file":
                    index_id, path = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type}
                elif node_type == "subject":
                    index_id, path, cmd = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type, "cmd": cmd}

        torch.save(node_to_path_type, out_file)
        connect.close()

    else:
        node_to_path_type = torch.load(out_file)

    return node_to_path_type
