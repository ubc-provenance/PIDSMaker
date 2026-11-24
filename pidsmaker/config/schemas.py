"""Dataset schemas: the entities, events and attributes of each family of datasets.

A schema describes what a dataset contains, independently of its domain (host provenance,
network authentication logs, network flows, ...):

- its node types: the database table of each type and the attributes read from it,
  which are used to build node labels (`construction.node_label_features`);
- its edge type vocabulary (`rel2id`), i.e. the categorical attribute of events stored in
  the `operation` column of the event table, along with the (source type, destination type)
  pairs where each edge type can appear (`possible_events`).

Each dataset of `DATASET_DEFAULT_CONFIG` refers to a schema by name (its `schema` key), and
the pipeline reads the database, builds the graphs and sizes the models from the schema,
instead of assuming the node and edge types of a particular dataset.

This module must not import torch nor other parts of the framework: it is imported by the
configuration.
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache

# Attribute values considered as null when building node labels
NULL_VALUES = frozenset({"None", "NA", "0", "none", "null", ""})


@dataclass(frozen=True)
class NodeTypeSchema:
    """A type of node, stored in its own database table.

    Attributes:
        name: name of the type, also used as key in `construction.node_label_features`.
        id: 1-based id of the type, used in one-hot encodings of node types.
        table: database table of the nodes of this type. All tables have the columns
            `node_uuid`, `hash_id` and `index_id`, plus the attribute columns.
        columns: (attribute, database column) pairs of the attributes of the nodes.
        auto_label: attributes used by `node_label_features=auto`. The label is the first
            non-null attribute of the list; if it is not the first one of the list, it is
            prefixed by `null_token` (with `construction.null_label_tokens`). When empty,
            all non-null attributes are concatenated in alphabetical order.
        null_token: token of a node without attribute (with `construction.null_label_tokens`).
        display: format of the path of a node shown in evaluation outputs, from its attributes.
        display_extra: (key, attribute) pairs of raw attributes added to evaluation outputs.
        label_format: how labels are tokenized: `path` (file system paths, split on '/'),
            `address` (IPs and ports, split on '.' and ':') or `name` (identifiers).
    """

    name: str
    id: int
    table: str
    columns: tuple
    auto_label: tuple = ()
    null_token: str = None
    display: str = None
    display_extra: tuple = ()
    label_format: str = "name"

    @property
    def attributes(self):
        return [attr for attr, _ in self.columns]

    def label_choices(self):
        """Values accepted by `construction.node_label_features.<type>`."""
        return ["auto", "type"] + self.attributes

    def auto_label_str(self, attrs, null_label_tokens):
        """Label of a node with `node_label_features=auto`, from its (cleaned) attributes."""
        if self.auto_label:
            for i, attr in enumerate(self.auto_label):
                value = attrs.get(attr, "")
                if value not in NULL_VALUES:
                    if i > 0 and null_label_tokens and self.null_token:
                        return f"{self.null_token} {value}"
                    return value
            return self.null_token if (null_label_tokens and self.null_token) else ""

        label = " ".join(
            attrs[key] for key in sorted(attrs) if key != "type" and attrs[key] not in NULL_VALUES
        )
        if not label and null_label_tokens and self.null_token:
            return self.null_token
        return label

    def display_path(self, raw_attrs):
        """Path shown for a node in evaluation outputs, from its raw database attributes."""
        fmt = self.display or "{" + self.attributes[0] + "}"
        return fmt.format(**{k: str(v) for k, v in raw_attrs.items()})


@dataclass(frozen=True)
class EdgeAttributeSchema:
    """An attribute of events, stored in a column of the event table.

    Attributes:
        name: name of the attribute, used in configs and as edge attribute of the graphs.
        column: column of the event table.
        kind: `categorical` (string values from `vocab`) or `numeric` (float values).
        vocab: values of a categorical attribute. Other values are mapped to `OTHER`.
    """

    name: str
    column: str
    kind: str
    vocab: tuple = ()

    OTHER = "OTHER"

    def __post_init__(self):
        if self.kind not in ("categorical", "numeric"):
            raise ValueError(f"Invalid kind {self.kind} of edge attribute {self.name}")
        if self.kind == "categorical" and not self.vocab:
            raise ValueError(f"Categorical edge attribute {self.name} needs a vocabulary")

    @property
    def values(self):
        """Values of a categorical attribute, including `OTHER`."""
        return list(self.vocab) + ([] if self.OTHER in self.vocab else [self.OTHER])

    @property
    def rel2id(self):
        """Bidirectional {value: id, id: value} vocabulary, with ids starting at 1."""
        mapping = {}
        for i, value in enumerate(self.values):
            mapping[i + 1] = value
            mapping[value] = i + 1
        return mapping

    def encode(self, value):
        """Value stored in the graphs, from the database value."""
        if self.kind == "numeric":
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
        value = str(value)
        return value if value in self.vocab else self.OTHER


@dataclass(frozen=True)
class DatasetSchema:
    """Description of the content of a family of datasets.

    Attributes:
        name: name of the schema, referenced by the `schema` key of datasets.
        node_types: node types, in the order in which their tables are read.
        rel2id: vocabulary of edge types of the graphs, as a bidirectional mapping
            {type: id, id: type} with ids starting at 1.
        possible_events: {(source type, destination type): [edge types]} of the
            (source, edge, destination) triplets that exist in the dataset.
        raw_rel2id: vocabulary of the `operation` column in the database, used to filter
            events, when it differs from `rel2id` (i.e. when edge types are translated).
        edge_type_translation: {(source type, raw edge type, destination type): edge type}
            translation applied to events when building the graphs. Untranslated edge types
            are kept as is.
        supports_edge_type_triplets: whether `edge_type_triplet` edge features are available.
        supports_hetero: whether heterogeneous encoders are available.
        event_table: database table of events (edges).
        edge_attributes: attributes of events besides their type, stored as edge attributes
            in the graphs and usable as edge features (`batching.edge_features=edge_attrs`).
        edge_type_attribute: name of the categorical edge attribute used as edge type, or
            None for the `operation` column (see `construction.edge_type`).
        label_column: column of the event table with the label of events (1: malicious),
            used as ground truth instead of node-level ground truth files. None if absent.
        operations: values of the `operation` column of the events kept when building graphs
            (None: all the edge types of the schema). The schema's value is the default, which
            `construction.operations` overrides (e.g. OpTC flows keep FLOW_START events by
            default, as LARES and Argus).
        noise_events_sql: SQL condition matching the events dropped with
            `construction.drop_noise_events` (e.g. broadcast and multicast flows), or None.
        drop_noise: whether the events of `noise_events_sql` are dropped.
    """

    name: str
    node_types: tuple
    rel2id: dict
    possible_events: dict
    raw_rel2id: dict = None
    edge_type_translation: dict = None
    supports_edge_type_triplets: bool = True
    supports_hetero: bool = True
    event_table: str = "event_table"
    edge_attributes: tuple = ()
    edge_type_attribute: str = None
    label_column: str = None
    operations: tuple = None
    noise_events_sql: str = None
    drop_noise: bool = False
    _by_name: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        ids = sorted(nt.id for nt in self.node_types)
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"Node type ids of schema {self.name} must be 1..N, found {ids}")
        object.__setattr__(self, "_by_name", {nt.name: nt for nt in self.node_types})

    @property
    def node_type_names(self):
        """Names of the node types, in id order."""
        return [nt.name for nt in sorted(self.node_types, key=lambda nt: nt.id)]

    @property
    def num_node_types(self):
        return len(self.node_types)

    @property
    def num_edge_types(self):
        return max(v for v in self.rel2id.values() if isinstance(v, int))

    @property
    def ntype2id(self):
        """Bidirectional mapping {type: id, id: type} with ids starting at 1."""
        mapping = {}
        for nt in sorted(self.node_types, key=lambda nt: nt.id):
            mapping[nt.id] = nt.name
            mapping[nt.name] = nt.id
        return mapping

    @property
    def raw_edge_types(self):
        """Values of the `operation` column of the database."""
        rel2id = self.raw_rel2id if self.raw_rel2id is not None else self.rel2id
        return [k for k in rel2id if isinstance(k, str)]

    @property
    def event_filter_rel2id(self):
        """Edge types of the database kept when building graphs."""
        rel2id = self.raw_rel2id if self.raw_rel2id is not None else self.rel2id
        if self.operations is None:
            return rel2id
        return {k: v for k, v in rel2id.items() if k in self.operations or v in self.operations}

    def operations_sql(self):
        """SQL condition (starting with ` AND`) selecting the events of `operations` (without
        the noise events with `drop_noise`), or ""."""
        sql = ""
        if self.operations is not None:
            values = ", ".join("'" + op.replace("'", "''") + "'" for op in self.operations)
            sql = f" AND operation IN ({values})"
        if self.drop_noise and self.noise_events_sql:
            sql += f" AND NOT ({self.noise_events_sql})"
        return sql

    def with_noise_dropped(self, drop):
        """The schema dropping its noise events (`construction.drop_noise_events`), if it has any."""
        if not drop or not self.noise_events_sql:
            return self
        return replace(self, drop_noise=True)

    def with_operations(self, operations):
        """The schema keeping the events of the given operations (`construction.operations`):
        `default` keeps the schema's default, `all` all events, otherwise a comma-separated
        list of values of the `operation` column."""
        if operations in (None, "default"):
            return self
        if operations == "all":
            return replace(self, operations=None)
        ops = tuple(op.strip() for op in operations.split(",") if op.strip())
        unknown = [op for op in ops if op not in self.raw_edge_types]
        if unknown or not ops:
            raise ValueError(
                f"Invalid construction.operations={operations} for schema {self.name}: "
                f"expected `default`, `all` or a comma-separated list of {self.raw_edge_types}"
            )
        return replace(self, operations=ops)

    def node_type(self, name):
        return self._by_name[name]

    def translate_edge_type(self, label, src_type, dst_type):
        if not self.edge_type_translation:
            return label
        return self.edge_type_translation.get((src_type, label, dst_type), label)

    def edge_attribute(self, name):
        for attr in self.edge_attributes:
            if attr.name == name:
                return attr
        raise ValueError(
            f"Unknown edge attribute {name} in schema {self.name}. "
            f"Available attributes: {[a.name for a in self.edge_attributes]}"
        )

    @property
    def edge_type_choices(self):
        """Values accepted by `construction.edge_type`."""
        return ["operation"] + [a.name for a in self.edge_attributes if a.kind == "categorical"]

    def with_edge_type(self, attribute):
        """The schema whose edge types are the values of a categorical edge attribute.

        Events are still filtered by their `operation`, but graphs and models use the values of
        the attribute as edge types. With `operation`, the schema is returned unchanged.
        """
        if attribute in (None, "operation"):
            return self
        attr = self.edge_attribute(attribute)
        if attr.kind != "categorical":
            raise ValueError(f"Edge attribute {attribute} is not categorical: it can't be an edge type")
        return replace(
            self,
            rel2id=attr.rel2id,
            possible_events={pair: list(attr.values) for pair in self.possible_events},
            raw_rel2id=self.event_filter_rel2id,
            edge_type_translation=None,
            edge_type_attribute=attribute,
        )


# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------

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

OPTC_hostname_map = {
    "optc_h051": "SysClient0051",
    "optc_h201": "SysClient0201",
    "optc_h501": "SysClient0501",
}


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------

# Node types of host provenance datasets (DARPA TC, OpTC, Carbon Black EDR), which share the
# same database tables. Tables are read in this order (netflows first), which defines the
# order of nodes in the outputs of the construction.
PROVENANCE_NODE_TYPES = (
    NodeTypeSchema(
        name="netflow",
        id=3,
        table="netflow_node_table",
        columns=(
            ("local_ip", "src_addr"),
            ("local_port", "src_port"),
            ("remote_ip", "dst_addr"),
            ("remote_port", "dst_port"),
        ),
        null_token="[NO_IP]",
        display="{local_ip}:{local_port}->{remote_ip}:{remote_port}",
        label_format="address",
    ),
    NodeTypeSchema(
        name="subject",
        id=1,
        table="subject_node_table",
        columns=(("path", "path"), ("cmd_line", "cmd")),
        auto_label=("cmd_line", "path"),
        null_token="[NO_CMD]",
        display="{path}",
        display_extra=(("cmd", "cmd_line"),),
        label_format="path",
    ),
    NodeTypeSchema(
        name="file",
        id=2,
        table="file_node_table",
        columns=(("path", "path"),),
        auto_label=("path",),
        null_token="[NO_PATH]",
        display="{path}",
        label_format="path",
    ),
)

SCHEMAS = {
    schema.name: schema
    for schema in [
        DatasetSchema(
            name="darpa_tc",
            node_types=PROVENANCE_NODE_TYPES,
            rel2id=rel2id_darpa_tc,
            possible_events=possible_events_darpa_tc,
        ),
        DatasetSchema(
            name="optc",
            node_types=PROVENANCE_NODE_TYPES,
            rel2id=rel2id_optc,
            possible_events=possible_events_optc,
            supports_edge_type_triplets=False,
            supports_hetero=False,
        ),
        # OpTC with edge types translated to their DARPA TC equivalent
        # (`construction.consistent_edge_types`), to share a vocabulary across datasets
        DatasetSchema(
            name="optc_tc_edge_types",
            node_types=PROVENANCE_NODE_TYPES,
            rel2id=rel2id_optc_consistent,
            possible_events=possible_events_optc_consistent,
            raw_rel2id=rel2id_optc,
            edge_type_translation=OPTC_TO_TC_EDGE_MAP,
            supports_edge_type_triplets=False,
            supports_hetero=False,
        ),
        DatasetSchema(
            name="carbon_black_edr",
            node_types=PROVENANCE_NODE_TYPES,
            rel2id=rel2id_graph_processor_carbon_black_edr,
            possible_events=possible_events_graph_processor_carbon_black_edr,
        ),
    ]
}

# --------------------------------------------------------------------------------------
# Network datasets
# --------------------------------------------------------------------------------------

# Hosts of enterprise networks (computers, IP addresses), identified by their name.
HOST_NODE_TYPE = NodeTypeSchema(
    name="host",
    id=1,
    table="host_node_table",
    columns=(("name", "name"),),
    auto_label=("name",),
    null_token="[NO_NAME]",
    display="{name}",
)

# LANL "Comprehensive, Multi-Source Cyber-Security Events" (Kent, 2015): authentication events
# between computers (auth.txt) and network flows between them (flows.txt), labelled by the red
# team events (redteam.txt). Authentications have the type (`operation`) of their orientation
# (LogOn, LogOff, ...), flows the type FLOW; attributes of the other kind of events are null.
# See `dataset_preprocessing/lanl/create_database_lanl.py`.
LANL_AUTH_ORIENTATIONS = ("LogOn", "LogOff", "TGS", "TGT", "AuthMap", "ScreenLock", "ScreenUnlock")
LANL_LOGON_TYPES = (
    "Network",
    "Service",
    "Batch",
    "Interactive",
    "RemoteInteractive",
    "CachedInteractive",
    "NetworkCleartext",
    "NewCredentials",
    "Unlock",
    "?",
)
LANL_AUTH_TYPES = ("NTLM", "Kerberos", "Negotiate", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0", "?")
# Kind of source user of authentications: user (U123@DOM1), computer account (C123$@DOM1),
# anonymous logon, or well-known system account (SYSTEM, LOCAL SERVICE, ...)
LANL_USER_KINDS = ("user", "computer", "anonymous", "system")

# Ports of flows: the most frequent well-known ports, and `N` for the de-identified ports of LANL
LANL_PORTS = (
    "N", "445", "80", "389", "137", "88", "139", "443", "135", "138", "123", "161", "1433", "22",
    "53", "3389",
)  # fmt: skip
LANL_PROTOCOLS = ("6", "17", "1", "41")  # TCP, UDP, ICMP, IPv6 encapsulation
LANL_EVENT_TYPES = LANL_AUTH_ORIENTATIONS + ("FLOW", "OTHER")

rel2id_lanl = {}
for _i, _event_type in enumerate(LANL_EVENT_TYPES):
    rel2id_lanl[_i + 1] = _event_type
    rel2id_lanl[_event_type] = _i + 1

LANL_EDGE_ATTRIBUTES = (
    # Authentications
    EdgeAttributeSchema("logon_type", "logon_type", "categorical", LANL_LOGON_TYPES),
    EdgeAttributeSchema("auth_type", "auth_type", "categorical", LANL_AUTH_TYPES),
    EdgeAttributeSchema("success", "success", "categorical", ("Success", "Fail")),
    EdgeAttributeSchema("src_user_kind", "src_user_kind", "categorical", LANL_USER_KINDS),
    EdgeAttributeSchema("dst_user_kind", "dst_user_kind", "categorical", LANL_USER_KINDS),
    # Flows
    EdgeAttributeSchema("src_port", "src_port", "categorical", LANL_PORTS),
    EdgeAttributeSchema("dst_port", "dst_port", "categorical", LANL_PORTS),
    EdgeAttributeSchema("protocol", "protocol", "categorical", LANL_PROTOCOLS),
    EdgeAttributeSchema("duration", "duration", "numeric"),
    EdgeAttributeSchema("packets", "packets", "numeric"),
    EdgeAttributeSchema("bytes", "bytes", "numeric"),
)

SCHEMAS["lanl_auth"] = DatasetSchema(
    name="lanl_auth",
    node_types=(HOST_NODE_TYPE,),
    rel2id=rel2id_lanl,
    possible_events={("host", "host"): list(LANL_EVENT_TYPES)},
    edge_attributes=LANL_EDGE_ATTRIBUTES,
    label_column="label",
)


# DARPA OpTC network flows of all hosts (FLOW START and FLOW MESSAGE events of eCAR), in the version
# preprocessed by LARES (Bilot et al., ACSAC 2026), whose hosts are integer ids. Ports are grouped as
# in LARES: common service ports are kept, other ports are grouped by range.
# See `dataset_preprocessing/optc/create_database_optc_net.py`.
rel2id_optc_flows = {}
for _i, _event_type in enumerate(("FLOW_START", "FLOW_MESSAGE", "OTHER")):
    rel2id_optc_flows[_i + 1] = _event_type
    rel2id_optc_flows[_event_type] = _i + 1

OPTC_PORTS = (
    "0", "8", "53", "67", "68", "80", "88", "135", "136", "137", "138", "139", "389", "443", "445",
    "1900", "3389", "5355", "5985", "1-1023", "1024-5354", "5356-50000", "50001-60000", "60001+",
)  # fmt: skip
OPTC_PROTOCOLS = ("6", "17", "1", "58")  # TCP, UDP, ICMP, ICMPv6

OPTC_FLOWS_EDGE_ATTRIBUTES = (
    EdgeAttributeSchema("src_port", "src_port", "categorical", OPTC_PORTS),
    EdgeAttributeSchema("dst_port", "dst_port", "categorical", OPTC_PORTS),
    EdgeAttributeSchema("protocol", "protocol", "categorical", OPTC_PROTOCOLS),
    EdgeAttributeSchema("size", "size", "numeric"),
)

SCHEMAS["optc_flows"] = DatasetSchema(
    name="optc_flows",
    node_types=(HOST_NODE_TYPE,),
    rel2id=rel2id_optc_flows,
    possible_events={("host", "host"): ["FLOW_START", "FLOW_MESSAGE", "OTHER"]},
    edge_attributes=OPTC_FLOWS_EDGE_ATTRIBUTES,
    label_column="label",
    # As in LARES and Argus, only FLOW START events are used by default
    operations=("FLOW_START",),
    # NetBIOS name/datagram (137, 138), LLMNR (5355) and SSDP (1900) broadcasts and multicasts
    # and ICMPv6 neighbor discovery (types 135, 136), received by every host: 99.86% of the
    # FLOW START events, they link each host to ~500 of the 625 hosts in every 30 min (LARES'
    # `filter_noise_protocols` drops 137, 138 and 5355)
    noise_events_sql=(
        "COALESCE(dst_port, '') IN ('137', '138', '5355', '1900') "
        "OR (COALESCE(protocol, '') = '58' AND COALESCE(dst_port, '') IN ('135', '136'))"
    ),
)


# Schema used instead of a dataset's schema with `construction.consistent_edge_types`
CONSISTENT_EDGE_TYPES_SCHEMAS = {"optc": "optc_tc_edge_types"}


def get_schema_by_name(name):
    if name not in SCHEMAS:
        raise ValueError(f"Unknown dataset schema {name}. Available schemas: {list(SCHEMAS)}")
    return SCHEMAS[name]


@lru_cache(maxsize=None)
def all_node_types():
    """{node type name: NodeTypeSchema} across all schemas (a name has the same attributes everywhere)."""
    node_types = {}
    for schema in SCHEMAS.values():
        for nt in schema.node_types:
            node_types.setdefault(nt.name, nt)
    return node_types


def get_schema(cfg):
    """Schema of the dataset of `cfg` (see the `schema` key of `DATASET_DEFAULT_CONFIG`),
    with the events selected by `construction.operations` and the edge types selected by
    `construction.edge_type`."""
    name = cfg.dataset.schema
    construction = getattr(cfg, "construction", None)
    if construction is not None and getattr(construction, "consistent_edge_types", False):
        name = CONSISTENT_EDGE_TYPES_SCHEMAS.get(name, name)
    schema = get_schema_by_name(name)
    edge_type = getattr(construction, "edge_type", None) if construction is not None else None
    operations = getattr(construction, "operations", None) if construction is not None else None
    return (
        schema.with_operations(operations)
        .with_noise_dropped(drops_noise_events(construction))
        .with_edge_type(edge_type)
    )


def drops_noise_events(construction):
    """`construction.drop_noise_events`, True when unset (configs not including default.yml)."""
    value = getattr(construction, "drop_noise_events", None) if construction is not None else None
    return True if value is None else bool(value)


def label_format(node_type):
    """Format of the labels of a node type (`path`, `address` or `name`), see `NodeTypeSchema`.
    None for texts that are not node labels (`node_type=None`)."""
    if node_type is None:
        return None
    node_types = all_node_types()
    if node_type not in node_types:
        raise ValueError(f"Invalid node type {node_type}")
    return node_types[node_type].label_format
