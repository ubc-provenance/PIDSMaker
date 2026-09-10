"""Ingests a provenance stream into a PIDSMaker dataset.

Real-time detection needs a trained model, and a model has to be trained on the
host it will watch. This entry point captures a stream (SPADE publishing to
Kafka, say) into the postgres schema the offline pipeline reads, so the eight
pipeline tasks can then train any PIDS on it unchanged:

    # 1. capture a benign period into a dataset - any name; MYHOST here
    python pidsmaker/stream_ingest.py MYHOST \\
        --stream_topic=benign --stream_idle_timeout=30

    # 2. train any system on it, keeping its weights for the detector
    python pidsmaker/main.py orthrus MYHOST \\
        --dataset_config=/home/artifacts/streaming/myhost/dataset.yml --save_model

    # 3. watch the live stream with it
    python pidsmaker/stream_detect.py orthrus MYHOST \\
        --dataset_config=... --stream_from_beginning=False

The dataset's dates are only known once the capture is over, so they are written
to a `dataset.yml` for steps 2 and 3 to pick up with `--dataset_config`.
"""

import argparse
import os
import signal
import sys
from datetime import datetime, timedelta

import yaml

from pidsmaker.config.config import DATASET_DEFAULT_CONFIG
from pidsmaker.config.pipeline import streamed_database_name
from pidsmaker.streaming.config import add_stream_args, build_stream, build_stream_cfg, str2bool
from pidsmaker.streaming.records import StreamEvent, StreamNode
from pidsmaker.streaming.sinks.postgres import PostgresSink
from pidsmaker.streaming.state import default_state_path, save_stream_state
from pidsmaker.utils.utils import log, ns_time_to_datetime_US

# The built-in dataset a streamed one inherits its conventions from, per producer.
TEMPLATE_BY_ADAPTER = {"spade": "SPADE_AUDIT"}


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest a provenance stream into a PIDSMaker dataset."
    )
    parser.add_argument(
        "dataset",
        help="Name of the dataset to create: any name you like for a streamed capture (its "
        "postgres database is the lowercased name), or a built-in one such as SPADE_AUDIT.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Built-in dataset whose conventions (edge vocabulary, node types) the new "
        "dataset inherits. Defaults to the one matching --stream_adapter (SPADE_AUDIT for spade).",
    )
    parser.add_argument("--database_host", default="postgres", help="Postgres host.")
    parser.add_argument("--database_user", default="postgres", help="Postgres user.")
    parser.add_argument("--database_password", default="postgres", help="Postgres password.")
    parser.add_argument("--database_port", type=int, default=5432, help="Postgres port.")
    parser.add_argument(
        "--append",
        type=str2bool,
        default=False,
        help="Add to the existing dataset instead of replacing it.",
    )
    parser.add_argument(
        "--artifact_dir", default="/home/artifacts/", help="Where the dataset config is written."
    )
    parser.add_argument(
        "--dataset_config_out",
        default=None,
        help="Path of the generated dataset config. Defaults to "
        "`<artifact_dir>/streaming/<database>/dataset.yml`.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Share of the captured days used for validation.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Share of the captured days used for testing.",
    )
    parser.add_argument(
        "--stream_state_out",
        default=None,
        help="Where to write what this run learned about nodes, so a detector joining the "
        "live stream later can resolve edges whose vertices were published during the "
        "capture. Defaults to `<artifact_dir>/streaming/<database>/stream_state.pkl`.",
    )
    parser.add_argument(
        "--log_every", type=int, default=50_000, help="Log progress every N records."
    )
    add_stream_args(parser)
    return parser.parse_args(argv)


def split_dates(dates, val_ratio: float, test_ratio: float):
    """Splits the captured days into train/val/test.

    PIDSMaker splits a dataset by calendar day, so a capture spanning a single day
    cannot be split at all. That is reported rather than silently accepted: the
    detection threshold would then come from the very data the model trained on.

    Args:
        dates: Sorted list of `YYYY-MM-DD` strings actually present in the capture.
        val_ratio, test_ratio: Share of days for validation and test.

    Returns:
        dict: `{"train_dates": [...], "val_dates": [...], "test_dates": [...]}`.
    """
    if len(dates) == 1:
        log(
            "Warning: the capture spans a single day, which cannot be split into train/val/test. "
            "Using it for all three: the validation threshold will be computed on training data, "
            "so the detector will be conservative. Capture at least three days for a sound split."
        )
        return {"train_dates": dates, "val_dates": dates, "test_dates": dates}

    if len(dates) == 2:
        log("Warning: only two days captured; validating on the training day.")
        return {"train_dates": dates[:1], "val_dates": dates[:1], "test_dates": dates[1:]}

    num_test = max(1, int(round(len(dates) * test_ratio)))
    num_val = max(1, int(round(len(dates) * val_ratio)))
    num_train = len(dates) - num_val - num_test
    if num_train < 1:  # very short captures: keep at least one training day
        num_train, num_val, num_test = len(dates) - 2, 1, 1

    return {
        "train_dates": dates[:num_train],
        "val_dates": dates[num_train : num_train + num_val],
        "test_dates": dates[num_train + num_val :],
    }


def write_dataset_config(args, sink, dates, path):
    """Writes the dataset config the pipeline needs to read this capture.

    Args:
        args: Parsed CLI arguments.
        sink: The postgres sink that just ingested the capture.
        dates: Sorted list of days present in the capture.
        path: Destination YAML file.

    Returns:
        dict: The written config.
    """
    splits = split_dates(dates, args.val_ratio, args.test_ratio)
    config = {
        # Everything the pipeline needs to know about a dataset that is not built in:
        # what to call it, where it lives, and which built-in dataset's conventions apply.
        "name": args.dataset,
        "template": args.template,
        "database": sink.database,
        "database_all_file": sink.database,
        "start_date": dates[0],
        # The pipeline expects `end_date` to fall after the last captured day.
        "end_date": (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
        "unused_dates": [],
        **splits,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    log(f"Dataset config written to {path}")
    log(
        f"  dataset: {config['name']} (database `{config['database']}`, "
        f"conventions of {config['template']})"
    )
    log(f"  train: {config['train_dates']}")
    log(f"  val:   {config['val_dates']}")
    log(f"  test:  {config['test_dates']}")
    return config


def main(argv=None):
    args = get_args(argv)

    if args.dataset in DATASET_DEFAULT_CONFIG:
        template = args.template or args.dataset
        database = DATASET_DEFAULT_CONFIG[args.dataset]["database"]
    else:
        template = args.template or TEMPLATE_BY_ADAPTER.get(args.stream_adapter, "SPADE_AUDIT")
        database = streamed_database_name(args.dataset)
    if template not in DATASET_DEFAULT_CONFIG:
        raise ValueError(
            f"Unknown template {template!r}. Built-in datasets: {sorted(DATASET_DEFAULT_CONFIG)}"
        )
    args.template = template

    stream_cfg = build_stream_cfg(args)
    stream = build_stream(stream_cfg)
    sink = PostgresSink(
        database=database,
        host=args.database_host,
        user=args.database_user,
        password=args.database_password,
        port=args.database_port,
        reset=not args.append,
    )

    # A capture is usually stopped by hand, and everything buffered until then must
    # still make it into the database.
    stopping = {"requested": False}

    def request_stop(signum, frame):
        if stopping["requested"]:
            sys.exit(1)
        stopping["requested"] = True
        log("Stopping ingestion, flushing buffers... (Ctrl-C again to abort)")
        if hasattr(stream.source, "stop"):
            stream.source.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    key_to_index = {}
    key_to_node = {}
    next_index = sink.max_existing_index + 1
    dates = set()

    log(f"Ingesting into database `{database}`...")
    try:
        for record in stream:
            if record is None:
                continue

            if isinstance(record, StreamNode):
                index_id = key_to_index.get(record.key)
                if index_id is None:
                    index_id = str(next_index)
                    next_index += 1
                    key_to_index[record.key] = index_id
                sink.write_node(record, index_id)
                key_to_node[record.key] = (record.node_type, record.attrs)

            elif isinstance(record, StreamEvent):
                src = key_to_index.get(record.src_key)
                dst = key_to_index.get(record.dst_key)
                if src is None or dst is None:
                    continue
                sink.write_event(record, src, dst)
                dates.add(ns_time_to_datetime_US(record.timestamp)[:10])

                if sink.num_events % args.log_every == 0:
                    log(
                        f"{sink.num_events} events, {sink.num_nodes} nodes ingested "
                        f"({stream.format_stats()})"
                    )
    finally:
        sink.flush()
        stream.close()

    sink.create_indices()
    log(f"Ingested {sink.num_events} events and {sink.num_nodes} nodes.")
    log(stream.format_stats())

    if sink.num_events == 0:
        log("No event was ingested: nothing to write a dataset config for.")
        sink.close()
        return

    config_path = args.dataset_config_out or os.path.join(
        args.artifact_dir, "streaming", database, "dataset.yml"
    )
    write_dataset_config(args, sink, sorted(dates), config_path)
    save_stream_state(
        args.stream_state_out or default_state_path(args.artifact_dir, database),
        stream.adapter,
        key_to_node,
    )
    sink.close()

    log("")
    log("Next step - train a system on this capture:")
    log(
        f"  python pidsmaker/main.py orthrus {args.dataset} "
        f"--dataset_config={config_path} --save_model"
    )


if __name__ == "__main__":
    main()
