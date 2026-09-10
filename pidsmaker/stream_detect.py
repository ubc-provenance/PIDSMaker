"""Runs a trained PIDS on a live provenance stream.

This is the real-time counterpart of `main.py`: instead of walking a dataset on
disk, it consumes provenance as it is produced (SPADE publishing to Kafka, a
capture file being tailed, ...), assembles it into time windows, scores each
window with a model trained by the normal pipeline, and reports the nodes whose
anomaly score crosses the detection threshold.

    python pidsmaker/stream_detect.py orthrus SPADE_AUDIT \\
        --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml \\
        --stream_brokers=kafka:9092 --stream_topic=spade-topic \\
        --stream_from_beginning=False --alert_sink=stdout,file \\
        --alert_file=/home/artifacts/streaming/alerts.jsonl

Any system the framework implements can be served this way, as long as its
featurizer can embed nodes it has never seen (see `pidsmaker/streaming/featurizer.py`).
"""

import argparse
import signal
import sys
import time

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.streaming.config import add_stream_args, build_stream, build_stream_cfg, str2bool
from pidsmaker.streaming.detector import StreamingDetector
from pidsmaker.streaming.graph_builder import StreamingGraphBuilder
from pidsmaker.streaming.records import StreamEvent, StreamNode
from pidsmaker.streaming.sinks.alerts import ALERT_SINKS, build_alert_sink
from pidsmaker.streaming.state import (
    apply_stream_state,
    default_state_path,
    load_stream_state,
    save_stream_state,
)
from pidsmaker.utils.utils import log


def add_detection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the streaming and alerting arguments to the shared pipeline parser."""
    add_stream_args(parser)

    group = parser.add_argument_group("detection")
    group.add_argument(
        "--alert_sink",
        default="stdout",
        help=f"Where detections go, comma-separated. Any of {ALERT_SINKS}.",
    )
    group.add_argument("--alert_file", default=None, help="Destination of the `file` alert sink.")
    group.add_argument("--alert_topic", default=None, help="Topic of the `kafka` alert sink.")
    group.add_argument(
        "--alert_log_all_windows",
        type=str2bool,
        default=False,
        help="Record windows with no alert too, making the alert file a full audit trail.",
    )
    group.add_argument(
        "--alert_threshold",
        type=float,
        default=None,
        help="Detection threshold. Defaults to the one the offline evaluation would use, "
        "computed from the validation losses of the trained run.",
    )
    group.add_argument(
        "--model_checkpoint",
        default=None,
        help="Trained model directory. Defaults to the `model_best` of the run matching "
        "this config.",
    )
    group.add_argument(
        "--stream_max_nodes",
        type=int,
        default=2_000_000,
        help="Node capacity of the batching tables. Must exceed the number of distinct "
        "nodes the stream will produce.",
    )
    group.add_argument(
        "--stream_max_events",
        type=int,
        default=2_000_000,
        help="How many past events stay available to the TGN neighbor loader before the "
        "temporal context is reset.",
    )
    group.add_argument(
        "--stream_state",
        default=None,
        help="Node knowledge written by `stream_ingest.py`, loaded so that edges whose "
        "vertices were published before this detector started can still be resolved. "
        "Defaults to `<artifact_dir>/streaming/<database>/stream_state.pkl`; pass `none` to "
        "start with no prior knowledge.",
    )
    group.add_argument(
        "--stream_state_save",
        type=str2bool,
        default=True,
        help="Write the (updated) node knowledge back when the detector stops, so a restart "
        "picks up where this run left off.",
    )
    group.add_argument(
        "--stream_stats_every",
        type=int,
        default=20,
        help="Log stream statistics every N time windows (0 to disable).",
    )
    group.add_argument(
        "--emit_viz",
        type=str2bool,
        default=False,
        help="Record every scored node (not just alerts) for the interactive web viewer: "
        "writes the embedding-viz artifacts plus a flat stream_scores.jsonl under a run "
        "directory the viewer discovers. Adds a dimensionality-reduction pass at shutdown.",
    )
    group.add_argument(
        "--viz_method",
        default="umap",
        help="Dimensionality-reduction method for --emit_viz (umap, tsne, pca).",
    )
    group.add_argument(
        "--viz_max_points",
        type=int,
        default=300_000,
        help="Cap on (node, window) snapshots kept for --emit_viz, to bound memory.",
    )
    return parser


def main(argv=None):
    args, unknown_args = get_runtime_required_args(
        return_unknown_args=True, args=argv, add_args_fn=add_detection_args
    )
    if unknown_args:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    stream_cfg = build_stream_cfg(args)

    stream = build_stream(stream_cfg)
    builder = StreamingGraphBuilder(
        cfg,
        window_size_minutes=stream_cfg.window_size,
        flush_timeout=stream_cfg.flush_timeout,
    )
    state_path = args.stream_state
    if state_path is None:
        state_path = default_state_path(cfg._artifact_dir, cfg.dataset.database)
    if state_path == "none":
        state_path = None

    if state_path:
        state = load_stream_state(state_path)
        if state is None:
            log(
                f"No stream state at {state_path}: edges whose vertices were published before "
                "this run started will be dropped. Consume from the beginning of the topic "
                "(`--stream_from_beginning=True`) to see those vertices."
            )
        else:
            apply_stream_state(state, stream.adapter, builder)

    viz_collector = None
    if args.emit_viz:
        from pidsmaker.streaming.viz import StreamingVizCollector

        viz_collector = StreamingVizCollector(
            cfg,
            artifact_dir=cfg._artifact_dir,
            method=args.viz_method,
            device=None,  # reduce_to_3d auto-detects GPU and falls back to CPU
            max_points=args.viz_max_points,
        )

    detector = StreamingDetector(
        cfg,
        threshold=args.alert_threshold,
        checkpoint_dir=args.model_checkpoint,
        max_nodes=args.stream_max_nodes,
        max_events=args.stream_max_events,
        viz_collector=viz_collector,
    )
    alert_sink = build_alert_sink(
        args.alert_sink,
        file_path=args.alert_file,
        brokers=stream_cfg.brokers,
        topic=args.alert_topic,
        backend=stream_cfg.backend,
        include_empty_windows=args.alert_log_all_windows,
    )

    # Say up front where everything this run produces will land, so the operator
    # never has to hunt for the alert file or the viz run afterwards.
    sinks = [s.strip() for s in str(args.alert_sink).split(",") if s.strip()]
    log("Outputs:")
    if "stdout" in sinks:
        log("  alerts → stdout (one summary line per window, plus each alert)")
    if "file" in sinks and args.alert_file:
        log(f"  alerts → {args.alert_file}")
    if "kafka" in sinks:
        log(f"  alerts → kafka topic '{args.alert_topic}' @ {stream_cfg.brokers}")
    if viz_collector is not None:
        log(f"  scores → {viz_collector.scores_path} (every scored node, per window)")
        log(f"  viz    → {viz_collector.run_dir} (open with pidsmaker.vizgen.web.viz_server)")
    if state_path and args.stream_state_save:
        log(f"  state  → {state_path} (node knowledge, for resuming)")
    window_min = builder.window_size_ns / 60_000_000_000  # effective size (config fallback applied)
    log(
        f"Window {window_min:g} min | flush {builder.flush_timeout:g}s idle | "
        f"threshold {detector.threshold:.4f} ({detector.threshold_method})"
    )

    stopping = {"requested": False}

    def request_stop(signum, frame):
        if stopping["requested"]:
            sys.exit(1)
        stopping["requested"] = True
        log("Stopping the detector, scoring the pending window... (Ctrl-C again to abort)")
        if hasattr(stream.source, "stop"):
            stream.source.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started = time.time()

    def handle_window(window):
        if window is None:
            return
        report = detector.process_window(window, builder.indexid2msg, builder.index_to_key)
        alert_sink.emit(report)
        # Offsets are only acknowledged once the window they cover has been scored,
        # so an interrupted detector re-reads what it had not yet judged.
        stream.commit()

        # A periodic roll-up (not per-window, to avoid flooding): cumulative volume
        # and throughput, plus a snapshot of the window just scored.
        if args.stream_stats_every and detector.num_windows % args.stream_stats_every == 0:
            elapsed = max(time.time() - started, 1e-9)
            eps = builder.num_events / elapsed
            log(
                f"[{detector.num_windows} windows | {builder.num_events} events, {eps:.0f}/s | "
                f"{detector.num_alerts} alerts] last {report.num_nodes}n/{report.num_edges}e, "
                f"max_score={report.max_score:.3f} vs thr {report.threshold:.3f}, "
                f"{report.inference_time * 1000:.0f}ms | {stream.format_stats()}"
            )

    log("Detector ready, waiting for provenance...")
    try:
        for record in stream:
            if record is None:
                handle_window(builder.flush_if_idle())
            elif isinstance(record, StreamNode):
                builder.add_node(record)
            elif isinstance(record, StreamEvent):
                handle_window(builder.add_event(record))
    finally:
        handle_window(builder.flush())
        stream.close()
        alert_sink.close()
        if state_path and args.stream_state_save:
            save_stream_state(state_path, stream.adapter, builder.node_state())

    elapsed = time.time() - started
    log(
        f"Detection finished: {builder.num_events} events, {detector.num_windows} windows, "
        f"{detector.num_alerts} alerts in {elapsed:.0f}s"
    )
    log(stream.format_stats())

    # Build the viewer last: the dimensionality reduction runs over every window at
    # once, so it belongs after the stream has drained, not on the scoring path.
    if viz_collector is not None:
        viz_collector.finalize()


if __name__ == "__main__":
    main()
