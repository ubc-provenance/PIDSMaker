"""Where real-time detections go.

A detection is only useful if it leaves the process: printed for an operator,
appended to a file another tool tails, or published back onto Kafka for a SIEM
to pick up. All three are the same interface, so `--alert_sink` swaps them.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from pidsmaker.utils.utils import log

ALERT_SINKS = ("stdout", "file", "kafka", "none")


@dataclass
class Alert:
    """One node flagged as anomalous within one time window."""

    node: int
    node_type: str
    label: str
    score: float
    threshold: float
    window_start: int
    window_end: int
    window: str
    window_events: int = 0
    # The identifier the producer gave this node (a SPADE vertex hash), so an alert
    # can be traced back to the capture it came from.
    source_key: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class WindowReport:
    """What the detector concluded about one time window."""

    window: str
    window_start: int
    window_end: int
    num_nodes: int
    num_edges: int
    num_events: int
    threshold: float
    max_score: float
    alerts: List[Alert] = field(default_factory=list)
    inference_time: float = 0.0


class AlertSink(ABC):
    """Publishes the detections of a time window."""

    @abstractmethod
    def emit(self, report: WindowReport) -> None:
        """Publishes one window's report."""

    def close(self) -> None:
        """Releases the sink's resources."""


class StdoutAlertSink(AlertSink):
    """Logs a one-line summary per window, then one line per alert.

    Args:
        max_alerts: Most alerts to print per window; the rest are summarized.
    """

    def __init__(self, max_alerts: int = 20):
        self.max_alerts = max_alerts

    def emit(self, report: WindowReport) -> None:
        log(
            f"[{report.window}] nodes={report.num_nodes} edges={report.num_edges} "
            f"max_score={report.max_score:.4f} thr={report.threshold:.4f} "
            f"alerts={len(report.alerts)} ({report.inference_time * 1000:.0f} ms)"
        )
        for alert in report.alerts[: self.max_alerts]:
            log(
                f"  ALERT node={alert.node} [{alert.node_type}] "
                f"score={alert.score:.4f} {alert.label}"
            )
        if len(report.alerts) > self.max_alerts:
            log(f"  ... and {len(report.alerts) - self.max_alerts} more alerts")


class FileAlertSink(AlertSink):
    """Appends one JSON object per alert to a file (JSON lines).

    Args:
        path: Destination file; parent directories are created.
        include_empty_windows: Also write a record for windows with no alert,
            which makes the file a full audit trail of what was scored.
    """

    def __init__(self, path: str, include_empty_windows: bool = False):
        self.path = path
        self.include_empty_windows = include_empty_windows
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "a", buffering=1)
        log(f"Writing alerts to {path}")

    def emit(self, report: WindowReport) -> None:
        for alert in report.alerts:
            self._file.write(alert.to_json() + "\n")
        if not report.alerts and self.include_empty_windows:
            self._file.write(
                json.dumps(
                    {
                        "window": report.window,
                        "alerts": 0,
                        "max_score": report.max_score,
                        "threshold": report.threshold,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def close(self) -> None:
        self._file.close()


class KafkaAlertSink(AlertSink):
    """Publishes alerts back onto a Kafka topic, as JSON.

    Args:
        brokers: Bootstrap servers.
        topic: Destination topic.
        backend: Kafka client to use (same choices as the source).
    """

    def __init__(self, brokers: str, topic: str, backend: str = "auto"):
        from pidsmaker.streaming.sources.kafka_source import _resolve_backend

        self.backend, module = _resolve_backend(backend)
        self.topic = topic
        if self.backend == "confluent":
            self._producer = module.Producer({"bootstrap.servers": brokers})
        else:
            self._producer = module.KafkaProducer(bootstrap_servers=brokers.split(","))
        log(f"Publishing alerts to Kafka topic {topic} on {brokers}")

    def emit(self, report: WindowReport) -> None:
        for alert in report.alerts:
            payload = alert.to_json().encode("utf-8")
            if self.backend == "confluent":
                self._producer.produce(self.topic, payload)
            else:
                self._producer.send(self.topic, payload)

    def close(self) -> None:
        self._producer.flush()
        if self.backend != "confluent":
            self._producer.close()


class NullAlertSink(AlertSink):
    """Discards everything. Useful for throughput measurements."""

    def emit(self, report: WindowReport) -> None:
        pass


class MultiAlertSink(AlertSink):
    """Fans a report out to several sinks."""

    def __init__(self, sinks: List[AlertSink]):
        self.sinks = sinks

    def emit(self, report: WindowReport) -> None:
        for sink in self.sinks:
            sink.emit(report)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def build_alert_sink(
    kinds: str,
    file_path: Optional[str] = None,
    brokers: Optional[str] = None,
    topic: Optional[str] = None,
    backend: str = "auto",
    include_empty_windows: bool = False,
) -> AlertSink:
    """Builds the alert sink(s) named by a comma-separated list.

    Args:
        kinds: Any combination of `ALERT_SINKS`, e.g. `"stdout,file"`.
        file_path: Destination of the `file` sink.
        brokers, topic, backend: Settings of the `kafka` sink.
        include_empty_windows: Passed to the `file` sink.

    Returns:
        AlertSink: A single sink, or a `MultiAlertSink` when several were asked for.
    """
    sinks = []
    for kind in [k.strip() for k in kinds.split(",") if k.strip()]:
        if kind == "stdout":
            sinks.append(StdoutAlertSink())
        elif kind == "file":
            if not file_path:
                raise ValueError("The `file` alert sink requires `--alert_file=<path>`.")
            sinks.append(FileAlertSink(file_path, include_empty_windows=include_empty_windows))
        elif kind == "kafka":
            if not brokers or not topic:
                raise ValueError("The `kafka` alert sink requires brokers and `--alert_topic`.")
            sinks.append(KafkaAlertSink(brokers, topic, backend=backend))
        elif kind == "none":
            sinks.append(NullAlertSink())
        else:
            raise ValueError(f"Invalid alert sink {kind!r}. Expected any of {ALERT_SINKS}.")

    if not sinks:
        raise ValueError("At least one alert sink is required.")
    return sinks[0] if len(sinks) == 1 else MultiAlertSink(sinks)
