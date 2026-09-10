"""Real-time provenance streaming for PIDSMaker.

Branches an external provenance capture agent onto the framework: records arrive
on a transport (Kafka), are decoded (Avro/JSON), translated by an adapter into
PIDSMaker's node/edge model, assembled into time windows, and either stored as a
dataset to train on (`pidsmaker/stream_ingest.py`) or scored live by a trained
PIDS (`pidsmaker/stream_detect.py`).

SPADE's Kafka storage is the first supported producer; adding another means
writing one adapter, see `pidsmaker/streaming/adapters/base.py`.
"""

from pidsmaker.streaming.records import StreamEvent, StreamNode

__all__ = ["StreamEvent", "StreamNode"]
