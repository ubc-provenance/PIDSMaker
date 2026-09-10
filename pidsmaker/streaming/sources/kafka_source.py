"""Kafka stream source.

This is the transport SPADE's Kafka storage publishes to, and the one any other
capture agent should use to feed PIDSMaker. Two client libraries are supported:
`confluent-kafka` (C client, preferred for throughput) and `kafka-python` (pure
Python fallback). The rest of the framework never sees which one is in use.
"""

from typing import Iterator, Optional

from pidsmaker.streaming.sources.base import StreamSource
from pidsmaker.utils.utils import log

BACKENDS = ("auto", "confluent", "kafka-python")


def _resolve_backend(backend: str):
    """Imports the requested Kafka client, falling back when `auto` is asked.

    Args:
        backend: One of `BACKENDS`.

    Returns:
        tuple: (backend name, imported module).
    """
    backend = backend.strip()
    errors = {}

    if backend in ("auto", "confluent"):
        try:
            import confluent_kafka

            return "confluent", confluent_kafka
        except ImportError as e:
            errors["confluent-kafka"] = e
            if backend == "confluent":
                raise

    if backend in ("auto", "kafka-python"):
        try:
            import kafka

            return "kafka-python", kafka
        except ImportError as e:
            errors["kafka-python"] = e
            if backend == "kafka-python":
                raise

    if backend not in BACKENDS:
        raise ValueError(f"Invalid Kafka backend {backend!r}. Expected one of {BACKENDS}.")

    raise ImportError(
        "No Kafka client available. Install one of `confluent-kafka` (recommended) or "
        f"`kafka-python`. Import errors: {errors}"
    )


class KafkaSource(StreamSource):
    """Consumes payloads from one or more Kafka topics.

    Args:
        brokers: Comma-separated `host:port` bootstrap servers.
        topics: Comma-separated topic list.
        group_id: Consumer group. Offsets are committed under this id, so
            restarting a detector with the same group resumes where it stopped.
        from_beginning: Whether a group with no committed offset starts at the
            oldest retained record (`True`) or only sees new ones (`False`).
        poll_timeout: Seconds to wait for a record before emitting an idle tick.
        backend: Which client library to use, see `BACKENDS`.
        max_records: Stop after this many payloads (0 = unlimited). Useful to
            bound ingestion runs and tests.
        idle_ticks_before_stop: Stop after this many consecutive idle ticks
            (0 = never stop). Used to drain a finite topic and exit.
        extra_config: Additional client properties, merged last so they can
            override anything set here.
    """

    def __init__(
        self,
        brokers: str,
        topics: str,
        group_id: str = "pidsmaker",
        from_beginning: bool = True,
        poll_timeout: float = 1.0,
        backend: str = "auto",
        max_records: int = 0,
        idle_ticks_before_stop: int = 0,
        extra_config: Optional[dict] = None,
    ):
        self.brokers = brokers
        self.topics = [t.strip() for t in topics.split(",") if t.strip()]
        if not self.topics:
            raise ValueError("At least one Kafka topic is required (`--stream.topic`).")
        self.group_id = group_id
        self.from_beginning = from_beginning
        self.poll_timeout = poll_timeout
        self.max_records = max_records
        self.idle_ticks_before_stop = idle_ticks_before_stop
        self.extra_config = extra_config or {}

        self.backend, self._module = _resolve_backend(backend)
        self._consumer = None
        self._stopped = False
        self._assigned = False
        self.num_records = 0

        log(
            f"Kafka source: {self.brokers} topics={self.topics} group={self.group_id} "
            f"({'from the beginning' if from_beginning else 'new records only'}) "
            f"using {self.backend} client"
        )

    def _build_consumer(self):
        offset_reset = "earliest" if self.from_beginning else "latest"

        if self.backend == "confluent":
            config = {
                "bootstrap.servers": self.brokers,
                "group.id": self.group_id,
                "auto.offset.reset": offset_reset,
                # Offsets are committed explicitly, once the records they cover have
                # actually been processed into a time window.
                "enable.auto.commit": False,
                **self.extra_config,
            }
            consumer = self._module.Consumer(config)

            def on_assign(_consumer, partitions):
                # Worth logging: with `from_beginning=False` the starting offset is
                # resolved at *assignment* time, not when the consumer subscribes, so
                # anything produced before this line is not seen by this run.
                self._assigned = True
                log(
                    "Kafka partitions assigned: "
                    + ", ".join(f"{p.topic}[{p.partition}]" for p in partitions)
                )

            consumer.subscribe(self.topics, on_assign=on_assign)
            return consumer

        consumer = self._module.KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.brokers.split(","),
            group_id=self.group_id,
            auto_offset_reset=offset_reset,
            enable_auto_commit=False,
            consumer_timeout_ms=int(self.poll_timeout * 1000),
            **self.extra_config,
        )
        return consumer

    def __iter__(self) -> Iterator[Optional[bytes]]:
        self._consumer = self._build_consumer()
        idle_ticks = 0

        while not self._stopped:
            payload = self._poll_one()

            if payload is None:
                idle_ticks += 1
                yield None
                if self.idle_ticks_before_stop and idle_ticks >= self.idle_ticks_before_stop:
                    log(f"Kafka source idle for {idle_ticks} polls, stopping.")
                    break
                continue

            idle_ticks = 0
            self.num_records += 1
            yield payload

            if self.max_records and self.num_records >= self.max_records:
                log(f"Reached max_records={self.max_records}, stopping.")
                break

    def _poll_one(self) -> Optional[bytes]:
        """Returns the next payload, or None when the poll timed out."""
        if self.backend == "confluent":
            message = self._consumer.poll(self.poll_timeout)
            if message is None:
                return None
            if message.error():
                # Partition EOF is not an error for us: it simply means the topic is
                # drained, which the idle-tick logic already handles.
                if message.error().code() == self._module.KafkaError._PARTITION_EOF:
                    return None
                raise RuntimeError(f"Kafka error: {message.error()}")
            return message.value()

        # kafka-python raises StopIteration on `consumer_timeout_ms`, which for us
        # is an idle tick rather than the end of the stream.
        try:
            message = next(self._consumer)
        except StopIteration:
            if not self._assigned and self._consumer.assignment():
                self._assigned = True
                log(f"Kafka partitions assigned: {sorted(self._consumer.assignment())}")
            return None
        if not self._assigned:
            self._assigned = True
            log(f"Kafka partitions assigned: {sorted(self._consumer.assignment())}")
        return message.value

    def commit(self) -> None:
        if self._consumer is None:
            return
        try:
            self._consumer.commit()
        except Exception as e:  # a failed commit only means records are re-read
            log(f"Warning: could not commit Kafka offsets: {e}")

    def stop(self) -> None:
        """Asks the iteration to end after the record being processed."""
        self._stopped = True

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
