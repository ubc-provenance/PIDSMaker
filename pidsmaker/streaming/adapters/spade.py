"""Adapter for SPADE's Kafka storage (`spade.storage.Kafka`).

SPADE publishes a stream of `GraphElement` records, each holding either a
`Vertex` or an `Edge` (see `cfg/spade.storage.Kafka.avsc` in the SPADE repo):

    Vertex: {annotations: map<string,string>, hash: string}
    Edge:   {annotations: map<string,string>, childVertexHash, parentVertexHash, hash}

Everything is carried in the annotations, using the OPM vocabulary SPADE's Audit
reporter emits (`spade.reporter.audit.OPMConstants`): vertices are `Process`,
`Artifact` or `Agent`, edges are `Used`, `WasGeneratedBy`, `WasTriggeredBy`,
`WasDerivedFrom` or `WasControlledBy` and carry an `operation` and a `time`.

Two translations happen here:

**Direction.** A SPADE edge points from the *effect* to the *cause*: `child` is
what happened, `parent` is what caused it (`Used(child=process, parent=file)` for
a read, `WasGeneratedBy(child=file, parent=process)` for a write). PIDSMaker
orients edges along the information flow instead, so `src` is always the parent
and `dst` always the child - which yields exactly the DARPA convention of
`file -> subject` for a read and `subject -> file` for a write.

**Vocabulary.** SPADE reports the full system call surface, optionally with a
qualifier (`"mmap (read)"`). Those operations are mapped onto the streamed
dataset's edge types (`rel2id_spade`); unmapped ones are dropped by default.
"""

import hashlib
from typing import Iterable, List, Optional, Union

from pidsmaker.streaming.adapters.base import ProvenanceAdapter
from pidsmaker.streaming.records import StreamEvent, StreamNode, parse_timestamp_to_ns
from pidsmaker.utils.dataset_utils import rel2id_spade

# SPADE annotation keys, from spade.reporter.audit.OPMConstants.
ANNOTATION_TYPE = "type"
ANNOTATION_SUBTYPE = "subtype"
ANNOTATION_OPERATION = "operation"
ANNOTATION_TIME = "time"
ANNOTATION_EVENT_ID = "event id"

VERTEX_PROCESS = "Process"
VERTEX_ARTIFACT = "Artifact"
VERTEX_AGENT = "Agent"

# Artifact subtypes that carry a remote endpoint and therefore become netflow nodes.
NETWORK_SUBTYPES = {"network socket", "network socket pair"}

# Annotations that identify *a version of* an artifact rather than the artifact
# itself. SPADE bumps them on every write when `versions=true`, which would make a
# single file look like a new node on every modification. They are excluded from
# the identity of a node when `merge_artifact_versions` is on.
VERSION_ANNOTATIONS = {
    "version",
    "epoch",
    "permissions",
    "size",
    "fd",
    "fd 0",
    "fd 1",
    "read fd",
    "write fd",
    "tgid",
    "pid",
    "event id",
    "time",
}

# SPADE operation -> streamed dataset edge type. Keys are either the operation
# itself, or a `(primary, secondary)` pair for the qualified operations SPADE
# builds as `"primary (secondary)"`.
OPERATION_TO_EVENT = {
    "read": "EVENT_READ",
    "write": "EVENT_WRITE",
    "open": "EVENT_OPEN",
    "close": "EVENT_CLOSE",
    "execve": "EVENT_EXECUTE",
    "fork": "EVENT_CLONE",
    "clone": "EVENT_CLONE",
    "unit": "EVENT_CLONE",
    "exit": "EVENT_EXIT",
    "connect": "EVENT_CONNECT",
    "accept": "EVENT_ACCEPT",
    "bind": "EVENT_BIND",
    "send": "EVENT_SENDTO",
    "sendto": "EVENT_SENDTO",
    "sendmsg": "EVENT_SENDTO",
    "recv": "EVENT_RECVFROM",
    "recvfrom": "EVENT_RECVFROM",
    "recvmsg": "EVENT_RECVFROM",
    "create": "EVENT_CREATE_OBJECT",
    "mknod": "EVENT_CREATE_OBJECT",
    "pipe": "EVENT_CREATE_OBJECT",
    "unlink": "EVENT_UNLINK",
    "rename": "EVENT_RENAME",
    "link": "EVENT_LINK",
    "mmap": "EVENT_MMAP",
    "mprotect": "EVENT_MPROTECT",
    "chmod": "EVENT_MODIFY_FILE_ATTRIBUTES",
    "setuid": "EVENT_CHANGE_PRINCIPAL",
    "setgid": "EVENT_CHANGE_PRINCIPAL",
    "kill": "EVENT_SIGNAL",
    "ptrace": "EVENT_MODIFY_PROCESS",
    "load": "EVENT_LOADLIBRARY",
    "init_module": "EVENT_LOADLIBRARY",
    "finit_module": "EVENT_LOADLIBRARY",
    "lseek": "EVENT_LSEEK",
    "dup": "EVENT_DUP",
    "truncate": "EVENT_TRUNCATE",
    "update": "EVENT_UPDATE",
    # Qualified operations: the copy itself is what matters, the qualifier only
    # says which side of the copy this edge represents.
    ("splice", "read"): "EVENT_READ",
    ("splice", "write"): "EVENT_WRITE",
    ("tee", "read"): "EVENT_READ",
    ("tee", "write"): "EVENT_WRITE",
    ("vmsplice", "read"): "EVENT_READ",
    ("vmsplice", "write"): "EVENT_WRITE",
    "splice": "EVENT_READ",
    "tee": "EVENT_READ",
    "vmsplice": "EVENT_WRITE",
}

UNMAPPED_EVENT = "EVENT_OTHER"


def parse_operation(operation: str):
    """Splits SPADE's `"primary (secondary)"` operation into its two parts.

    Args:
        operation: Raw `operation` annotation.

    Returns:
        tuple: (primary, secondary or None), both lowercased and stripped.
    """
    operation = (operation or "").strip().lower()
    if operation.endswith(")") and "(" in operation:
        primary, secondary = operation[:-1].split("(", 1)
        return primary.strip(), secondary.strip()
    return operation, None


class SpadeKafkaAdapter(ProvenanceAdapter):
    """Translates SPADE Kafka-storage records into PIDSMaker nodes and events.

    Args:
        merge_artifact_versions: Treat every version of an artifact as the same
            node. SPADE's `versions=true` emits a fresh vertex (and hash) per
            write, which would otherwise turn one file into hundreds of nodes
            that share no history.
        keep_unmapped_operations: Emit `EVENT_OTHER` for operations absent from
            `OPERATION_TO_EVENT` instead of dropping those edges.
        keep_agents: Keep `WasControlledBy` edges to `Agent` vertices. Off by
            default: agents are users/groups, which have no PIDSMaker node type.
        drop_self_loops: Drop edges whose endpoints collapsed onto the same node
            (what artifact-version edges become once versions are merged).
        subject_path_from: Which annotation supplies a process' `path`, in order
            of preference.
    """

    name = "spade"
    default_format = "avro"
    default_schema = "spade.storage.Kafka"

    def __init__(
        self,
        merge_artifact_versions: bool = True,
        keep_unmapped_operations: bool = False,
        keep_agents: bool = False,
        drop_self_loops: bool = True,
        subject_path_from: str = "exe,name,command line",
        **options,
    ):
        super().__init__(**options)
        self.merge_artifact_versions = merge_artifact_versions
        self.keep_unmapped_operations = keep_unmapped_operations
        self.keep_agents = keep_agents
        self.drop_self_loops = drop_self_loops
        self.subject_path_from = [f.strip() for f in subject_path_from.split(",") if f.strip()]

        # SPADE always publishes a vertex before the edges that reference it, so
        # resolving an endpoint hash to the node key it collapsed onto is a plain
        # lookup. Hashes of vertices we deliberately skipped (agents) map to None.
        self.hash_to_key = {}

    def edge_type_vocabulary(self) -> Optional[dict]:
        return rel2id_spade

    def handle(self, record) -> Iterable[Union[StreamNode, StreamEvent]]:
        element = _unwrap_element(record)
        if element is None:
            self.stats["skipped_malformed"] += 1
            return []

        kind, payload = element
        annotations = _unwrap_map(payload.get("annotations")) or {}

        if kind == "edge":
            return self._handle_edge(payload, annotations)
        return self._handle_vertex(payload, annotations)

    def _handle_vertex(self, payload: dict, annotations: dict) -> List[StreamNode]:
        vertex_hash = _unwrap_union(payload.get("hash"))
        if not vertex_hash:
            self.stats["skipped_malformed"] += 1
            return []

        vertex_type = annotations.get(ANNOTATION_TYPE, "")

        if vertex_type == VERTEX_PROCESS:
            node_type = "subject"
            attrs = {
                "path": _first_present(annotations, self.subject_path_from),
                "cmd_line": annotations.get("command line", ""),
            }
        elif vertex_type == VERTEX_ARTIFACT:
            subtype = annotations.get(ANNOTATION_SUBTYPE, "")
            if subtype in NETWORK_SUBTYPES:
                node_type = "netflow"
                attrs = {
                    "local_ip": annotations.get("local address", ""),
                    "local_port": annotations.get("local port", ""),
                    "remote_ip": annotations.get("remote address", ""),
                    "remote_port": annotations.get("remote port", ""),
                }
            else:
                node_type = "file"
                path = _artifact_path(annotations, subtype)
                if not annotations.get("path"):
                    # A descriptor-only artifact: SPADE never saw the `open` that
                    # named it (an fd inherited from before the capture, a pipe, a
                    # socket pair). This counts vertex *payloads*, so on a real
                    # capture it is inflated by SPADE's per-write versioning - the
                    # same fd churns thousands of versions that merge back to one
                    # node. A high count means the capture started mid-flight, not
                    # that most nodes are anonymous.
                    self.stats["payload_file_no_path"] += 1
                attrs = {"path": path}
        else:
            # Agents (users/groups) have no PIDSMaker node type. Recording the hash
            # as unmapped keeps the edges that reference them cheap to drop.
            self.stats[f"skipped_vertex_{vertex_type or 'unknown'}"] += 1
            self.hash_to_key[vertex_hash] = None
            return []

        key = self._node_key(vertex_hash, node_type, annotations)
        self.hash_to_key[vertex_hash] = key
        self.stats[f"node_{node_type}"] += 1
        return [StreamNode(key=key, node_type=node_type, attrs=attrs)]

    def _node_key(self, vertex_hash: str, node_type: str, annotations: dict) -> str:
        """Returns the key a vertex' node is stored under.

        Processes keep their own hash: two processes that share a path are genuinely
        different nodes. Artifacts optionally collapse onto their identity
        annotations so that every version of a file is one node.
        """
        if not self.merge_artifact_versions or node_type == "subject":
            return vertex_hash

        identity = sorted((k, v) for k, v in annotations.items() if k not in VERSION_ANNOTATIONS)
        digest = hashlib.md5(repr(identity).encode("utf-8")).hexdigest()
        return f"{node_type}:{digest}"

    def _handle_edge(self, payload: dict, annotations: dict) -> List[StreamEvent]:
        child = _unwrap_union(payload.get("childVertexHash"))
        parent = _unwrap_union(payload.get("parentVertexHash"))
        if not child or not parent:
            self.stats["skipped_edge_missing_endpoint"] += 1
            return []

        # SPADE edges point effect -> cause; information flows the other way.
        src_key = self.hash_to_key.get(parent, _MISSING)
        dst_key = self.hash_to_key.get(child, _MISSING)

        if src_key is _MISSING or dst_key is _MISSING:
            # The vertex was never published (a topic consumed from the middle, or a
            # SPADE restart). Nothing sound can be built from a dangling endpoint.
            self.stats["skipped_edge_unknown_endpoint"] += 1
            return []
        if src_key is None or dst_key is None:
            self.stats["skipped_edge_to_agent"] += 1
            return []
        if self.drop_self_loops and src_key == dst_key:
            # `update` links one version of an artifact to the next. With versions
            # merged those are the same node, so the edge is version bookkeeping
            # rather than a real self-loop - worth counting separately, because a
            # capture full of them means `merge_artifact_versions` is doing its job.
            if annotations.get(ANNOTATION_OPERATION) == "update":
                self.stats["skipped_edge_version_update"] += 1
            else:
                self.stats["skipped_edge_self_loop"] += 1
            return []

        primary, secondary = parse_operation(annotations.get(ANNOTATION_OPERATION, ""))
        event_type = OPERATION_TO_EVENT.get((primary, secondary)) or OPERATION_TO_EVENT.get(primary)
        if event_type is None:
            if not self.keep_unmapped_operations:
                self.stats[f"skipped_operation_{primary or 'missing'}"] += 1
                return []
            event_type = UNMAPPED_EVENT

        timestamp = parse_timestamp_to_ns(annotations.get(ANNOTATION_TIME))
        if timestamp is None:
            self.stats["skipped_edge_missing_time"] += 1
            return []

        edge_hash = _unwrap_union(payload.get("hash")) or ""
        event_id = annotations.get(ANNOTATION_EVENT_ID, "")
        self.stats[f"event_{event_type}"] += 1

        return [
            StreamEvent(
                key=edge_hash or f"{parent}-{child}-{event_id}",
                src_key=src_key,
                dst_key=dst_key,
                operation=event_type,
                timestamp=timestamp,
            )
        ]


# Sentinel telling "this hash was never seen" apart from "this hash was seen and
# deliberately not turned into a node".
_MISSING = object()


def _first_present(annotations: dict, keys: List[str]) -> str:
    for key in keys:
        value = annotations.get(key)
        if value:
            return value
    return ""


def _artifact_path(annotations: dict, subtype: str) -> str:
    """Builds the `path` of a non-network artifact.

    Files and directories have a real path. The others (memory regions, pipes,
    message queues, ...) do not, so they get a synthetic one built from whatever
    identifies them, keeping node labels meaningful for the text featurizers.
    """
    path = annotations.get("path")
    if path:
        return path
    for key in ("memory address", "id", "root path"):
        value = annotations.get(key)
        if value:
            return f"<{subtype or 'unknown'}:{value}>"
    return f"<{subtype or 'unknown'}>"


def _unwrap_union(value):
    """Unwraps Avro's JSON union encoding (`{"string": "x"}` -> `"x"`)."""
    if isinstance(value, dict) and len(value) == 1:
        key = next(iter(value))
        if key in ("string", "null", "bytes"):
            return value[key]
    return value


def _unwrap_map(value):
    """Unwraps Avro's JSON map encoding (`{"map": {...}}` -> `{...}`)."""
    value = _unwrap_union(value)
    if isinstance(value, dict) and set(value) == {"map"}:
        return value["map"]
    return value


def _unwrap_element(record):
    """Locates the vertex or edge inside a decoded `GraphElement`.

    Handles every shape the record can arrive in: fastavro's plain dict, the
    `(record name, value)` tuples fastavro produces with `return_record_name`, and
    the raw Avro-JSON encoding with its union tag (`{"spade.storage.kafka.Edge": ...}`)
    when a plain JSON decoder was used.

    Returns:
        tuple: `("edge" | "vertex", payload dict)`, or None if the record holds neither.
    """
    if not isinstance(record, dict):
        return None

    element = record.get("element", record)

    if isinstance(element, (tuple, list)) and len(element) == 2 and isinstance(element[0], str):
        name, payload = element
        if isinstance(payload, dict):
            return ("edge" if name.endswith("Edge") else "vertex"), payload
        return None

    if isinstance(element, dict) and len(element) == 1:
        name = next(iter(element))
        if name.startswith("spade.storage.kafka."):
            payload = element[name]
            if isinstance(payload, dict):
                return ("edge" if name.endswith("Edge") else "vertex"), payload
            return None

    if not isinstance(element, dict) or "hash" not in element:
        return None

    # Classified on the endpoints' *values* rather than on the presence of their
    # keys: a decoder that materializes the whole union (fastavro does, since both
    # endpoint fields default to null) gives a vertex those keys set to null.
    is_edge = (
        _unwrap_union(element.get("childVertexHash")) is not None
        or _unwrap_union(element.get("parentVertexHash")) is not None
    )
    return ("edge" if is_edge else "vertex"), element
