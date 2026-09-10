"""Stream sources: where raw provenance payloads come from.

A source knows nothing about schemas or provenance semantics; it only produces
payloads (and idle ticks). Kafka is the source used to branch SPADE onto
PIDSMaker, but replaying a file behaves identically, which is what makes the
whole path testable without a broker.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class StreamSource(ABC):
    """Yields raw payloads from a provenance stream.

    Iterating a source yields `bytes` for each record, and `None` as an *idle
    tick* whenever the source has been waiting without receiving anything. Ticks
    let the consumer make progress on a quiet host (flush a partial time window,
    log throughput) instead of blocking until the next event.

    Sources that read a self-describing container (an Avro object file, say) have
    already decoded their records by the time they can hand them over. They set
    `yields_records = True` and yield dicts instead of bytes; the pipeline then
    skips the decoding step.
    """

    yields_records = False

    @abstractmethod
    def __iter__(self) -> Iterator[Optional[bytes]]:
        """Yields payloads (bytes) and idle ticks (None)."""

    def commit(self) -> None:
        """Acknowledges the payloads consumed so far, when the source supports it."""

    def close(self) -> None:
        """Releases the source's resources."""
