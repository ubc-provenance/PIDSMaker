"""CLI arguments and factory for the streaming layer.

Streaming is not one of the eight pipeline tasks: it feeds them (ingestion) or
consumes their output (real-time detection), so its settings live outside the
task config and never take part in the task hashes. They are plain
`--stream_*` CLI arguments, collected here so the two streaming entry points
(`stream_ingest.py`, `stream_detect.py`) expose exactly the same knobs.
"""

import argparse
import os
import time

from yacs.config import CfgNode as CN

from pidsmaker.streaming.adapters import build_adapter, get_adapter_class
from pidsmaker.streaming.decoders import FORMATS, build_decoder
from pidsmaker.streaming.sources import FileSource, KafkaSource
from pidsmaker.streaming.sources.file_source import FILE_FORMATS
from pidsmaker.streaming.sources.kafka_source import BACKENDS
from pidsmaker.streaming.stream import ProvenanceStream

SOURCES = ("kafka", "file")


def str2bool(value):
    """Parses a boolean CLI value, using the same `True`/`False` spelling as the pipeline args."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def add_stream_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds every `--stream_*` argument to a parser.

    Args:
        parser: The parser to extend.

    Returns:
        The same parser, for chaining.
    """
    group = parser.add_argument_group("streaming")

    group.add_argument(
        "--stream_source",
        default="kafka",
        choices=SOURCES,
        help="Where provenance comes from: a Kafka topic, or a capture file to replay.",
    )
    group.add_argument(
        "--stream_adapter",
        default="spade",
        help="Which producer's schema the stream carries (see pidsmaker/streaming/adapters).",
    )
    group.add_argument(
        "--stream_format",
        default=None,
        choices=list(FORMATS) + list(FILE_FORMATS),
        help="Wire format of the payloads. Defaults to the adapter's own format "
        "(`avro` for SPADE's Kafka storage).",
    )
    group.add_argument(
        "--stream_schema",
        default=None,
        help="Avro schema: a path to a .avsc file, or the name of one bundled in "
        "pidsmaker/streaming/schemas. Defaults to the adapter's own schema.",
    )

    # Kafka source
    group.add_argument(
        "--stream_brokers", default="kafka:9092", help="Kafka bootstrap servers (host:port,...)."
    )
    group.add_argument(
        "--stream_topic", default="spade-topic", help="Kafka topic(s), comma-separated."
    )
    group.add_argument(
        "--stream_group_id",
        default=None,
        help="Kafka consumer group. By default every run gets a new group, so "
        "`--stream_from_beginning=True` always replays the whole topic and `False` only "
        "sees records produced from now on. Name a group to resume where the previous run "
        "with that name stopped.",
    )
    group.add_argument(
        "--stream_from_beginning",
        type=str2bool,
        default=True,
        help="Read the topic from its oldest record (True) or only from now on (False). "
        "With a named `--stream_group_id` that already committed offsets, those win.",
    )
    group.add_argument(
        "--stream_backend",
        default="auto",
        choices=BACKENDS,
        help="Kafka client library to use.",
    )
    group.add_argument(
        "--stream_poll_timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a record before emitting an idle tick.",
    )

    # File source
    group.add_argument(
        "--stream_file", default=None, help="Capture file to replay (with `--stream_source=file`)."
    )
    group.add_argument(
        "--stream_follow",
        type=str2bool,
        default=False,
        help="Keep reading the capture file as it grows, instead of stopping at its end.",
    )
    group.add_argument(
        "--stream_rate",
        type=float,
        default=0.0,
        help="Replay speed in records/second (0 = as fast as possible).",
    )

    # Run bounds
    group.add_argument(
        "--stream_max_records",
        type=int,
        default=0,
        help="Stop after this many records (0 = run until interrupted).",
    )
    group.add_argument(
        "--stream_idle_timeout",
        type=float,
        default=0.0,
        help="Stop after this many seconds without a single record (0 = never stop). "
        "Set it to drain a finite topic and exit.",
    )
    group.add_argument(
        "--stream_on_error",
        default="skip",
        choices=["skip", "raise"],
        help="What to do with a record that fails to decode or translate.",
    )

    # Windowing
    group.add_argument(
        "--stream_window_size",
        type=float,
        default=None,
        help="Time window size in minutes. Defaults to `construction.time_window_size`, "
        "which is what the model was trained with.",
    )
    group.add_argument(
        "--stream_flush_timeout",
        type=float,
        default=60.0,
        help="Close a pending time window after this many seconds of wall clock, so a quiet "
        "host still gets scored (0 = only close windows on their time boundary).",
    )

    # Adapter options
    group.add_argument(
        "--stream_merge_artifact_versions",
        type=str2bool,
        default=True,
        help="SPADE adapter: treat all versions of an artifact as a single node.",
    )
    group.add_argument(
        "--stream_keep_unmapped_operations",
        type=str2bool,
        default=False,
        help="SPADE adapter: keep operations with no edge type as `EVENT_OTHER` "
        "instead of dropping them.",
    )
    group.add_argument(
        "--stream_keep_agents",
        type=str2bool,
        default=False,
        help="SPADE adapter: keep edges pointing to Agent (user/group) vertices.",
    )

    return parser


def build_stream_cfg(args) -> CN:
    """Collects the `--stream_*` args into a config node.

    Args:
        args: Parsed CLI arguments.

    Returns:
        CfgNode: The streaming configuration, with the adapter's defaults filled in.
    """
    adapter_cls = get_adapter_class(args.stream_adapter)

    cfg = CN()
    cfg.source = args.stream_source
    cfg.adapter = args.stream_adapter
    # Same producer, different container: SPADE's Kafka *server* writer emits bare
    # Avro records, while its *file* writer wraps them in an Avro object container.
    default_format = adapter_cls.default_format
    if args.stream_source == "file" and default_format == "avro":
        default_format = "avro_container"
    cfg.format = args.stream_format or default_format
    cfg.schema = args.stream_schema or adapter_cls.default_schema
    cfg.brokers = args.stream_brokers
    cfg.topic = args.stream_topic
    # A fresh group per run unless one is named: what `--stream_from_beginning` says is
    # then what happens, instead of silently resuming from an earlier run's offsets.
    cfg.group_id = args.stream_group_id or f"pidsmaker-{int(time.time())}-{os.getpid()}"
    cfg.from_beginning = args.stream_from_beginning
    cfg.backend = args.stream_backend
    cfg.poll_timeout = args.stream_poll_timeout
    cfg.file = args.stream_file
    cfg.follow = args.stream_follow
    cfg.rate = args.stream_rate
    cfg.max_records = args.stream_max_records
    cfg.idle_timeout = args.stream_idle_timeout
    cfg.on_error = args.stream_on_error
    cfg.window_size = args.stream_window_size
    cfg.flush_timeout = args.stream_flush_timeout
    cfg.adapter_options = CN()
    cfg.adapter_options.merge_artifact_versions = args.stream_merge_artifact_versions
    cfg.adapter_options.keep_unmapped_operations = args.stream_keep_unmapped_operations
    cfg.adapter_options.keep_agents = args.stream_keep_agents
    return cfg


def build_stream(stream_cfg: CN) -> ProvenanceStream:
    """Builds the source/decoder/adapter chain described by a streaming config.

    Args:
        stream_cfg: Output of `build_stream_cfg()`.

    Returns:
        ProvenanceStream: The iterator of canonical records.
    """
    # An idle timeout is expressed in seconds by the user but enforced as a number
    # of consecutive empty polls, which is what both sources actually count.
    idle_ticks = 0
    if stream_cfg.idle_timeout:
        idle_ticks = max(
            1, int(round(stream_cfg.idle_timeout / max(stream_cfg.poll_timeout, 1e-6)))
        )

    if stream_cfg.source == "kafka":
        source = KafkaSource(
            brokers=stream_cfg.brokers,
            topics=stream_cfg.topic,
            group_id=stream_cfg.group_id,
            from_beginning=stream_cfg.from_beginning,
            poll_timeout=stream_cfg.poll_timeout,
            backend=stream_cfg.backend,
            max_records=stream_cfg.max_records,
            idle_ticks_before_stop=idle_ticks,
        )
    elif stream_cfg.source == "file":
        if not stream_cfg.file:
            raise ValueError("`--stream_source=file` requires `--stream_file=<path>`.")
        source = FileSource(
            path=stream_cfg.file,
            fmt=stream_cfg.format,
            follow=stream_cfg.follow,
            poll_timeout=stream_cfg.poll_timeout,
            rate=stream_cfg.rate,
            max_records=stream_cfg.max_records,
            idle_ticks_before_stop=idle_ticks,
        )
    else:
        raise ValueError(f"Invalid stream source {stream_cfg.source!r}. Expected one of {SOURCES}.")

    decoder = (
        None
        if getattr(source, "yields_records", False)
        else build_decoder(stream_cfg.format, stream_cfg.schema)
    )
    adapter = build_adapter(stream_cfg.adapter, **dict(stream_cfg.adapter_options))

    return ProvenanceStream(
        source=source, decoder=decoder, adapter=adapter, on_error=stream_cfg.on_error
    )
