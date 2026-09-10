"""Wires a source, a decoder and an adapter into one iterator of canonical records.

    bytes from Kafka/file -> decoder (Avro/JSON) -> adapter (SPADE/...) -> canonical records

Each of the three stages is swappable, which is what makes branching a new
provenance producer onto PIDSMaker a matter of adding one adapter rather than
touching the ingestion or detection code.
"""

from collections import Counter
from typing import Iterator, Optional, Union

from pidsmaker.streaming.adapters.base import ProvenanceAdapter
from pidsmaker.streaming.decoders import RecordDecoder
from pidsmaker.streaming.records import StreamEvent, StreamNode
from pidsmaker.streaming.sources.base import StreamSource
from pidsmaker.utils.utils import log


class ProvenanceStream:
    """Yields canonical records from a raw stream.

    Iterating yields `StreamNode`/`StreamEvent` objects, and `None` as an idle
    tick propagated from the source so callers can flush pending work.

    Args:
        source: Where payloads come from.
        decoder: How to turn a payload into a record. May be None when the
            source already yields decoded records (`source.yields_records`).
        adapter: How to turn a record into canonical nodes and events.
        on_error: `"skip"` counts and ignores records that fail to decode or
            translate, `"raise"` propagates the exception. Skipping is the
            default: one corrupt record must not take a live detector down.
    """

    def __init__(
        self,
        source: StreamSource,
        decoder: Optional[RecordDecoder],
        adapter: ProvenanceAdapter,
        on_error: str = "skip",
    ):
        if decoder is None and not getattr(source, "yields_records", False):
            raise ValueError("A decoder is required unless the source yields decoded records.")
        self.source = source
        self.decoder = decoder
        self.adapter = adapter
        self.on_error = on_error
        self.stats = Counter()

    def __iter__(self) -> Iterator[Optional[Union[StreamNode, StreamEvent]]]:
        source_yields_records = getattr(self.source, "yields_records", False)

        for payload in self.source:
            if payload is None:
                yield None
                continue

            self.stats["payloads"] += 1
            try:
                record = payload if source_yields_records else self.decoder.decode(payload)
                if record is None:
                    self.stats["empty_payloads"] += 1
                    continue
                results = self.adapter.handle(record)
            except Exception as e:
                self.stats["errors"] += 1
                if self.on_error == "raise":
                    raise
                if self.stats["errors"] <= 10:
                    log(f"Warning: dropping unreadable record ({type(e).__name__}: {e})")
                continue

            for result in results:
                self.stats["nodes" if isinstance(result, StreamNode) else "events"] += 1
                yield result

    def commit(self) -> None:
        """Acknowledges everything consumed so far."""
        self.source.commit()

    def close(self) -> None:
        """Releases the underlying source."""
        self.source.close()

    def format_stats(self) -> str:
        """One-line summary of what the stream has produced and dropped."""
        adapter_stats = ", ".join(f"{k}={v}" for k, v in sorted(self.adapter.stats.items()))
        stream_stats = ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items()))
        return f"[stream] {stream_stats} | [adapter] {adapter_stats}"
