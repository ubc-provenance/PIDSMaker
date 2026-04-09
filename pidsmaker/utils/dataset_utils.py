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

# TODO: add possible events for other datasets (i.e., with different edges)

ntype2id = {
    1: "subject",
    "subject": 1,
    2: "file",
    "file": 2,
    3: "netflow",
    "netflow": 3,
}

optc_datasets = {"optc_h201", "optc_h501", "optc_h051"}
atlasv2_datasets = {"atlasv2_h1"}
graph_processor_carbon_black_edr_datasets = {"atlasv2_edr", "carbanakv2_edr"}

optc_hostname_map = {
    "optc_h051": "SysClient0051",
    "optc_h201": "SysClient0201",
    "optc_h501": "SysClient0501",
}

def decrement_dict(d):
    return {
        k - 1 if isinstance(k, int) else k: v - 1 if isinstance(v, int) else v for k, v in d.items()
    }


def get_rel2id(cfg, from_zero=False):
    dataset_name = cfg.dataset.name.lower()

    if dataset_name in optc_datasets:
        return decrement_dict(rel2id_optc) if from_zero else rel2id_optc
    elif dataset_name in atlasv2_datasets:
        return rel2id_atlasv2
    elif dataset_name in graph_processor_carbon_black_edr_datasets:
        return decrement_dict(rel2id_graph_processor_carbon_black_edr) if from_zero else rel2id_graph_processor_carbon_black_edr
    else:
        return decrement_dict(rel2id_darpa_tc) if from_zero else rel2id_darpa_tc


def get_node_map(from_zero=False):
    if from_zero:
        return decrement_dict(ntype2id)
    return ntype2id


def get_num_edge_type(cfg):
    dataset_name = cfg.dataset.name.lower()

    if dataset_name not in optc_datasets and "edge_type_triplet" in cfg.batching.edge_features:
        possible_events = get_possible_events(cfg)
        return sum([len(events) for events in possible_events.values()])
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
