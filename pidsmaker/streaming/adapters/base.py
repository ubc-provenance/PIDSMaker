"""Provenance adapters: from a source's own schema to PIDSMaker's node/edge model.

An adapter is the *only* place that knows what a given capture agent's records
mean. It receives decoded records one by one and emits `StreamNode` /
`StreamEvent` objects (see `pidsmaker.streaming.records`). Everything downstream
- windowing, featurization, batching, the detector, the postgres sink - is
source-agnostic.

To branch a new provenance source onto PIDSMaker:
    1. Subclass `ProvenanceAdapter` and implement `handle()`.
    2. Register it in `pidsmaker.streaming.adapters.ADAPTERS`.
    3. Point `--stream.adapter` at it.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Optional, Union

from pidsmaker.streaming.records import StreamEvent, StreamNode


class ProvenanceAdapter(ABC):
    """Translates one source's records into canonical nodes and events.

    Attributes:
        name: Identifier used by `--stream.adapter`.
        default_format: Wire format the source publishes by default, used when
            `--stream.format` is left unset.
        default_schema: Avro schema shipped for that format, if any.
        stats: Counter of what happened to the records seen so far (kept for
            observability: how many were nodes, events, or dropped and why).
    """

    name: str = "base"
    default_format: str = "json"
    default_schema: Optional[str] = None

    def __init__(self, **options):
        self.options = options
        self.stats = Counter()

    @abstractmethod
    def handle(self, record) -> Iterable[Union[StreamNode, StreamEvent]]:
        """Translates one decoded record.

        Args:
            record: A decoded record (usually a dict) from the stream.

        Returns:
            Iterable of `StreamNode` and/or `StreamEvent`. Empty when the record
            carries nothing usable (an agent vertex, an unmapped operation, ...);
            such cases should be counted in `self.stats` rather than raising, so
            that a single odd record never takes the detector down.
        """

    def edge_type_vocabulary(self) -> Optional[dict]:
        """Returns the `rel2id`-style vocabulary this adapter emits, if it defines one."""
        return None
