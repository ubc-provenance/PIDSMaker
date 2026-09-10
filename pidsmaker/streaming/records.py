"""Canonical provenance records exchanged between stream adapters and the rest of the framework.

Every streaming source (SPADE's Kafka storage today, auditd/CamFlow/EDR agents
tomorrow) speaks its own schema. Adapters translate that schema into the two
records defined here, which are the *only* thing the graph builder, the postgres
sink and the real-time detector ever see. Adding a new provenance source
therefore means writing one adapter, not touching the pipeline.

The model is intentionally the same one the offline pipeline already uses:
three node types (`subject`, `file`, `netflow`) and typed, timestamped edges
oriented along the *information flow* direction (src -> dst).
"""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional

# The node types the framework knows about. Kept in sync with
# `pidsmaker.utils.dataset_utils.ntype2id`.
NODE_TYPES = ("subject", "file", "netflow")

# The raw attributes each node type carries. These are exactly the columns of the
# postgres node tables, so a streamed dataset is indistinguishable from a
# pre-processed DARPA one once ingested.
NODE_ATTRIBUTES = {
    "subject": ("path", "cmd_line"),
    "file": ("path",),
    "netflow": ("local_ip", "local_port", "remote_ip", "remote_port"),
}


@dataclass
class StreamNode:
    """A provenance entity (process, file or network flow).

    Attributes:
        key: Source-specific stable identifier (e.g. SPADE's vertex hash). Used to
            resolve the endpoints of events and to deduplicate re-emitted nodes.
        node_type: One of `NODE_TYPES`.
        attrs: Raw attributes, restricted to the keys in `NODE_ATTRIBUTES`.
            Missing attributes default to an empty string downstream.
    """

    key: str
    node_type: str
    attrs: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type {self.node_type}, expected one of {NODE_TYPES}")


@dataclass
class StreamEvent:
    """A provenance event (an edge) between two entities.

    `src_key`/`dst_key` follow the information-flow convention used everywhere in
    PIDSMaker: a read is `file -> subject`, a write is `subject -> file`. Adapters
    are responsible for reorienting their source's edges accordingly.

    Attributes:
        key: Source-specific event identifier (SPADE's edge hash), stored as
            `event_uuid`.
        src_key: Key of the source node.
        dst_key: Key of the destination node.
        operation: Edge type, already mapped to the dataset's vocabulary
            (e.g. `EVENT_READ`).
        timestamp: Event time in nanoseconds since epoch.
    """

    key: str
    src_key: str
    dst_key: str
    operation: str
    timestamp: int


def parse_timestamp_to_ns(value, default: Optional[int] = None) -> Optional[int]:
    """Converts a source timestamp to nanoseconds since epoch.

    Sources report time in wildly different units: SPADE's Audit reporter emits
    fractional epoch seconds (`"1756456789.123"`), other producers emit integer
    milliseconds, microseconds or nanoseconds. The magnitude of the value is
    unambiguous enough (any plausible capture happens after 2001) to pick the
    right unit without asking the user to configure it.

    Args:
        value: The raw timestamp (str, int or float).
        default: Returned when the value is missing or unparsable.

    Returns:
        int: Nanoseconds since epoch, or `default`.
    """
    if value is None:
        return default
    try:
        # Decimal rather than float: an epoch time in nanoseconds needs 19
        # significant digits, and a float64 only carries about 16 - enough to shift
        # "1756456789.123" by nearly 200 microseconds.
        as_decimal = Decimal(str(value).strip())
    except (TypeError, ValueError, InvalidOperation):
        return default
    if as_decimal <= 0:
        return default

    # Thresholds are one order of magnitude apart, so a value can only fall in one
    # bucket: < 1e11 is seconds (until year 5138), < 1e14 milliseconds, etc.
    if as_decimal < Decimal("1e11"):
        return int(as_decimal * 1_000_000_000)
    if as_decimal < Decimal("1e14"):
        return int(as_decimal * 1_000_000)
    if as_decimal < Decimal("1e17"):
        return int(as_decimal * 1_000)
    return int(as_decimal)
