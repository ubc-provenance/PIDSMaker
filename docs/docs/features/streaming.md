## Real-time detection from a live provenance stream

PIDSMaker can be fed provenance **as it is produced**, instead of from a pre-processed
database. A capture agent publishes provenance to Kafka, PIDSMaker consumes it, assembles
it into the same time window graphs the offline pipeline builds, and scores each window
with a trained PIDS as soon as it closes.

!!! tip "In a hurry?"
    [The tutorial](streaming_tutorial.md) walks the whole thing end to end — SPADE to
    capture to training to live alerts — with the output of every command. This page is
    the reference behind it.

[SPADE](https://github.com/ashish-gehani/SPADE) is the first supported producer, through
its **Kafka storage**. Its schema is deliberately simple — a stream of vertices and edges
carrying annotations — which is why it is the right place to branch onto: SPADE's Audit
reporter already turns Linux Audit records into that shape, and any other agent
(auditd, CamFlow, an EDR) can be added by writing a single adapter.

```
   host under watch                          PIDSMaker
┌────────────────────┐   Kafka    ┌──────────────────────────────────┐
│ Linux Audit        │   topic    │ source → decoder → adapter       │
│   ↓                │  ───────►  │      ↓                           │
│ SPADE Audit        │  (Avro)    │ time window graphs               │
│   reporter         │            │      ↓                           │
│   ↓                │            │ online featurization → batching  │
│ SPADE Kafka        │            │      ↓                           │
│   storage          │            │ trained PIDS → alerts            │
└────────────────────┘            └──────────────────────────────────┘
```

### The three stages, and why they are separate

| Stage | Question it answers | Swap it with |
|---|---|---|
| **Source** | Where do payloads come from? | `--stream_source=kafka` / `file` |
| **Decoder** | What encoding are they in? | `--stream_format=avro` / `avro_json` / `json` |
| **Adapter** | What do they *mean*? | `--stream_adapter=spade` |

Only the adapter knows a producer's schema. It emits `StreamNode` and `StreamEvent`
records (`pidsmaker/streaming/records.py`), and everything downstream — windowing,
featurization, batching, detection, the postgres sink — is producer-agnostic.

## Setting up SPADE

Build SPADE from its [`pidsmaker-dev` branch](https://github.com/ashish-gehani/SPADE/tree/pidsmaker-dev)
with a JDK 21 or newer (the [tutorial](streaming_tutorial.md#2-build-spade) has the exact
steps). Its Kafka storage takes the broker and the topic as arguments, so nothing in its
`cfg/` needs editing:

```sh
bin/spade start
printf 'add storage Kafka kafka.output.server=localhost:29092 kafka.output.topic=benign\nexit\n' | bin/spade control
printf 'add reporter Audit fileIO=true netIO=true\nexit\n' | bin/spade control      # live capture; needs root
```

!!! tip "Let the capture script do this"
    `scripts/capture/new_capture.sh --topic NAME` is the capture agent: it starts SPADE if
    needed, attaches its Kafka storage to the topic you name, records the host's activity
    through the Linux audit subsystem and streams it into SPADE for as long as it runs,
    with every precondition checked first. Ctrl-C stops it. The three steps below are
    what you run against that topic, from another terminal. The tutorial is built around it.

SPADE's `vagrant/kafka` and `vagrant/trace` directories automate a similar setup, including
an `audit-to-kafka` scenario that replays an existing auditd log.

!!! tip "Two SPADE settings worth changing"
    `cfg/spade.reporter.Audit.config` ships with `fileIO=false` and `netIO=false`, which
    suppress `read`/`write` and `send`/`recv` edges — the bulk of the data flow a PIDS
    learns from. Set both to `true` before capturing anything you intend to train on.

!!! warning "Vertex deduplication"
    SPADE's default `Deduplicate` screen (`cfg/spade.core.AbstractStorage.config`)
    publishes each vertex once. A detector that joins the topic later therefore receives
    edges whose endpoints it has never seen and has to drop them. PIDSMaker solves this by
    reloading the node knowledge the ingestion run saved (see
    [Joining a stream in progress](#joining-a-stream-in-progress)); the alternative is to
    consume the topic from the beginning, or to remove that screen so every vertex is
    republished.

A broker is all PIDSMaker needs on the transport side. A single-node one is enough:

```sh
docker compose -f compose-kafka.yml up -d
```

## Three steps to a live detector

A model has to be trained on the host it will watch, so a capture comes first.

### 1. Capture a benign period into a dataset

`stream_ingest.py` writes the stream into the postgres schema the offline pipeline reads,
so a SPADE capture becomes a dataset indistinguishable from a pre-processed DARPA one.
The dataset can have any name: `stream_ingest.py MYHOST …` creates database `myhost` and
a `dataset.yml` that names the built-in dataset it inherits its conventions from
(`template: SPADE_AUDIT` for the SPADE adapter; `--template` overrides it). Every later
command takes that name plus `--dataset_config=<that file>`.

```sh
python pidsmaker/stream_ingest.py SPADE_AUDIT \
    --stream_brokers=kafka:9092 --stream_topic=benign \
    --stream_idle_timeout=30
```

It reads the whole topic, stops once it has been quiet for `--stream_idle_timeout`
seconds (or on Ctrl-C, flushing what it holds), and writes two files next to the artifacts:

- `dataset.yml` — the dates the capture actually covers, split into train/val/test.
  A dataset's split is by calendar day, so **capture at least three days**; with fewer,
  ingestion warns and validates on the training day.
- `stream_state.pkl` — what it learned about nodes, used in step 3.

### 2. Train any system on the capture

```sh
python pidsmaker/main.py orthrus SPADE_AUDIT \
    --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml --save_model
```

`--save_model` writes the weights the detector will load (`training/<hash>/trained_models/model_best`).
The whole pipeline runs unchanged. The `evaluation` task reports that a streamed capture
has no labelled attacks and stops there — detection metrics need positives — after
`training` has saved the model and the validation losses the threshold comes from.

### 3. Score the live stream

```sh
python pidsmaker/stream_detect.py orthrus SPADE_AUDIT \
    --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml \
    --stream_brokers=kafka:9092 --stream_topic=incident \
    --stream_from_beginning=False \
    --alert_sink=stdout,file --alert_file=/home/artifacts/streaming/alerts.jsonl
```

`--stream_from_beginning` decides what "the stream" is: `True` scores everything the topic
already holds (a recording made earlier, drained with `--stream_idle_timeout`), `False`
only what is published from now on (a detector left running while the host is in use).

Every closed window is scored and summarized, and nodes above the threshold are reported:

```
[2026-08-28 12:30:10~2026-08-28 12:43:48] nodes=91 edges=147 max_score=3.4400 thr=3.4392 alerts=8 (12 ms)
  ALERT node=1735 [subject] score=3.4400 subject /usr/bin/curl curl https://example.com
```

The same command works against a capture file instead of a broker, which is the easiest
way to rehearse a detection run:

```sh
python pidsmaker/stream_detect.py orthrus SPADE_AUDIT --dataset_config=... \
    --stream_source=file --stream_file=/tmp/kafka-output.json --stream_format=avro_json
```

## How a live window is scored

Everything the offline pipeline does per time window happens here too, with the same
functions, so a live batch is indistinguishable from a pre-computed one:

1. **Windowing** — events accumulate until one arrives past `construction.time_window_size`
   minutes from the window's start. On a quiet host, `--stream_flush_timeout` closes a
   pending window anyway, so a detector never sits on half a window indefinitely.
2. **Online featurization** — the node embeddings come from the featurization model the
   training run saved. This works because those models are *inductive over text*: a path
   never seen during training still gets a vector.
3. **Batching** — the same global/intra/inter batching the config asks for, with the
   batching state (the TGN last-neighbor graph, the node caches) kept alive across windows.
4. **Detection** — the trained model scores the window, node scores are reduced exactly as
   `node_evaluation` does, and compared against the threshold the offline evaluation would
   have used, derived from the run's validation losses (`--alert_threshold` overrides it).

!!! note "Latency of a live capture"
    SPADE's audit bridge holds the newest 10,000 audit events in a reordering buffer and
    emits an event only once 10,000 newer ones have arrived, so provenance reaches the
    topic with that lag — seconds on a busy host, longer on a quiet one — and the last
    events of a capture are published when the capture stops. A detector on the topic sees
    the same lag; its windows close on the *event* timestamps, so nothing is mis-windowed.

!!! note "One unavoidable difference"
    Offline, a node's score is its maximum over the whole test set. A detector cannot know
    the future, so a node is scored within each window, as soon as that window closes.

### Featurizers that can be served online

| Method | Online | Why |
|---|---|---|
| `word2vec`, `fasttext`, `doc2vec`, `hierarchical_hashing` | ✅ | Embed a node from its own label |
| `only_type`, `only_ones` | ✅ | No embedding at all |
| `flash` | ✅ | Embeds from the node's own events, accumulated per window |
| `temporal_rw`, `alacarte`, `ocrapt_features` | ❌ | Need random walks or dataset-wide statistics, which a live stream cannot provide |

A system trained with one of the last three raises a clear error rather than silently
producing wrong vectors.

## Joining a stream in progress

A detector started after the capture — the normal case, since training happens in between —
misses the vertices already published. `stream_detect.py` therefore loads
`stream_state.pkl` (written by `stream_ingest.py`, and refreshed when the detector stops),
which maps the producer's vertex identifiers to the nodes they were folded into:

```
Restored 1519 nodes and 1519 vertex ids from the stream state: edges whose vertices
were published before this run started can still be resolved.
```

Pass `--stream_state=none` to start with no prior knowledge, or
`--stream_from_beginning=True` to replay the topic instead.

!!! note "Offsets and consumer groups"
    Every run consumes under a **new Kafka consumer group** unless `--stream_group_id`
    names one, so `--stream_from_beginning` means exactly what it says on every run: the
    whole topic, or only new records. Name a group to get Kafka's resume semantics
    instead: offsets are committed only after the window they cover has been scored, so
    a detector restarted with the same group re-reads what it had not yet judged and
    nothing else.

## Where alerts go

`--alert_sink` takes a comma-separated list:

| Sink | Output |
|---|---|
| `stdout` | A line per window and per alert |
| `file` | One JSON object per alert (`--alert_file`), appended |
| `kafka` | The same JSON published to `--alert_topic`, for a SIEM to pick up |
| `none` | Discards everything, for throughput measurements |

### Recording every score for analysis and the web viewer

The alert sinks only carry what crossed the threshold. `--emit_viz` additionally records
*every* scored node — the whole distribution — at shutdown, under a run directory the
embedding viewer discovers (`<ARTIFACTS_DIR>/detection/evaluation/stream_<timestamp>/<dataset>/`):

- `stream_scores.jsonl` — one row per node per window (`window`, `node`, `node_type`,
  `label`, `score`, `is_alert`, `source_key`), for pandas/SQL analysis.
- `viz/embedding_viz_<dataset>_word2vec_points.json` (+ `_adj.json`) — the same artifacts
  the offline pipeline writes, so `python -m pidsmaker.vizgen.web.viz_server` plays the run
  back with temporal scrub, per-node inspection and graph overlays.

A dimensionality-reduction pass runs at shutdown, so `--emit_viz` adds time and memory to
the end of a run; `--viz_method` (umap/tsne) and `--viz_max_points` tune it. See the
[tutorial](streaming_tutorial.md#visualizing-and-analysing-the-predictions) for a worked
example.

## Adding another provenance source

Everything except the adapter is already source-agnostic. To branch a new producer:

1. Subclass `ProvenanceAdapter` in `pidsmaker/streaming/adapters/`, implementing
   `handle(record)` so it yields `StreamNode` / `StreamEvent` objects. Two things are
   yours to get right: mapping the producer's entity types onto `subject`/`file`/`netflow`,
   and orienting edges along the **information flow** (`src` → `dst`: a read is
   `file → subject`, a write is `subject → file`).
2. Register it in `ADAPTERS` (`pidsmaker/streaming/adapters/__init__.py`).
3. If its vocabulary differs, add a `rel2id_*` to `pidsmaker/utils/dataset_utils.py` and a
   dataset entry declaring `num_edge_types`.

Then `--stream_adapter=<name>` is all that changes; the source, decoder, ingestion,
detection and alerting stay as they are. If the producer also publishes to Kafka with Avro,
nothing else needs writing at all.

### What the SPADE adapter does

Two translations, both worth knowing about when reading its output:

**Direction.** A SPADE edge points from the *effect* to the *cause*
(`Used(child=process, parent=file)` for a read). PIDSMaker orients edges along the
information flow, so `src` is always the parent and `dst` the child — which is exactly the
DARPA convention of `file → subject` for a read.

**Vocabulary.** SPADE reports the full system call surface, sometimes qualified
(`"mmap (read)"`). Those operations are mapped onto the 28 edge types of `rel2id_spade`;
unmapped ones are dropped unless `--stream_keep_unmapped_operations=True`.

Two behaviours are configurable because they are judgement calls:

- `--stream_merge_artifact_versions` (default on) treats every version of a file as one
  node. SPADE's `versions=true` emits a fresh vertex per write, which would otherwise turn
  a single file into hundreds of nodes sharing no history.
- `--stream_keep_agents` (default off) keeps edges to `Agent` (user/group) vertices, which
  have no PIDSMaker node type.

## Arguments

Both entry points share the `--stream_*` arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--stream_source` | `kafka` | `kafka` or `file` |
| `--stream_adapter` | `spade` | Which producer's schema the stream carries |
| `--stream_format` | adapter's | `avro`, `avro_json`, `json`, `avro_container` |
| `--stream_schema` | adapter's | `.avsc` path, or a name bundled in `pidsmaker/streaming/schemas` |
| `--stream_brokers` | `kafka:9092` | Bootstrap servers |
| `--stream_topic` | `spade-topic` | Topic(s), comma-separated |
| `--stream_group_id` | a new group per run | Consumer group; name one to resume from its last committed offset |
| `--stream_from_beginning` | `True` | Read the whole topic (`True`) or only records published from now on (`False`) |
| `--stream_idle_timeout` | `0` | Stop after N seconds without a record (0 = run until interrupted) |
| `--stream_max_records` | `0` | Stop after N records |
| `--stream_file`, `--stream_follow`, `--stream_rate` | — | Replaying a capture file, optionally tailing it or throttling it |
| `--stream_window_size` | `construction.time_window_size` | Window size in minutes |
| `--stream_flush_timeout` | `60` | Close a pending window after N seconds of wall clock |
| `--stream_on_error` | `skip` | `skip` or `raise` on an unreadable record |

`stream_detect.py` adds `--alert_sink`, `--alert_file`, `--alert_topic`,
`--alert_threshold`, `--model_checkpoint`, `--stream_state`, `--stream_max_nodes` (node
capacity of the batching tables) and `--stream_max_events` (how much history stays
available to the TGN neighbor loader). For analysis and the web viewer it also takes
`--emit_viz` (record every scored node, not just alerts), `--viz_method` (umap/tsne) and
`--viz_max_points` (cap on retained node-snapshots).

## Limitations

- **Detection quality is a property of the trained model**, not of the transport. A system
  trained on a short or unrepresentative capture will alert accordingly.
- **`magic`'s threshold** is computed from the test set's own embedding distances, which a
  live stream has no equivalent of. Pass `--alert_threshold` to run it.
- **Capacity is bounded**: `--stream_max_nodes` sizes the batching tables, and reaching
  `--stream_max_events` resets the TGN temporal context rather than growing without bound.
- **A single stream is a single host.** Watching several hosts means one detector per
  topic, or an adapter that partitions by host.
