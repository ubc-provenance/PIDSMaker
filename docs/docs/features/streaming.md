# Real-time detection

PIDSMaker can be fed provenance **as it is produced**, instead of from a pre-processed
database. A capture agent on a host publishes its provenance to Kafka, PIDSMaker assembles
the stream into the same time window graphs the offline pipeline builds, and a trained
system scores each window as soon as it closes.

This page first walks from an empty machine to a model scoring provenance produced by
your own machine's system calls, **as it happens** — about half an hour, most of which is
SPADE compiling and the model training. The second half explains how it works, lists every
argument, and shows how to add another provenance source.

[SPADE](https://github.com/ashish-gehani/SPADE) is the supported capture agent: its Audit
reporter turns Linux Audit records into provenance and its Kafka storage publishes them.
Any other producer (auditd, CamFlow, an EDR) can be added by writing a single adapter, see
[Adding another provenance source](#adding-another-provenance-source).

```
  host                                                        PIDSMaker container
┌─────────────────────────────────────────────┐  Kafka    ┌──────────────────────────────┐
│ scripts/capture/new_capture.sh --topic NAME │  topic    │ stream_ingest.py ─▶ dataset  │
│                                             │  NAME     │ main.py          ─▶ model    │
│ Linux audit ─▶ SPADE ───────────────────────┼─────────▶ │                              │
│ (records the syscalls) (turns them into     │  (live)   │ stream_detect.py ─▶ alerts,  │
│                         provenance)         │           │                    web viewer│
└─────────────────────────────────────────────┘           └──────────────────────────────┘
```

Two things to keep in mind throughout:

- **Where a command runs.** The capture agent runs **on the host**, because the audit
  subsystem is the host's and reading it needs `sudo`. Everything else — building a dataset,
  training, detecting, the web viewer — runs **in the PIDSMaker container**, which is what
  the `docker compose -f compose-pidsmaker.yml exec pids python ...` prefix does.
- **The topic is the handle.** A Kafka topic is a named, durable log on the broker. The
  capture agent appends the host's provenance to the topic you name, and every PIDSMaker
  command reads it by that name (`--stream_topic=NAME`). Reuse a name to keep adding to it;
  pick a new one (or `--fresh`) to start over.

You will:

1. bring up postgres, Kafka and the PIDSMaker container;
2. build SPADE;
3. capture a benign period into a topic, turn it into a dataset, and train a model on it;
4. capture again while the real-time detector watches the topic, and stage an intrusion;
5. explore every score in the web viewer.

!!! warning "What is real here, and what is not"
    Every count and timestamp on this page comes from a real run: real Linux syscalls,
    recorded by the kernel audit subsystem, turned into provenance by SPADE's real Audit
    reporter, scored by a real model. Where a block
    comes from a longer capture than the one shown next to it, it says so. What is
    *staged* is the intrusion in [step 4](#4-detect-in-real-time): it is built from
    ordinary binaries and decoy data (a fake credentials file, a loopback "C2" listener)
    so that running it is harmless. Short captures prove the pipeline, not detection
    quality; [step 6](#6-reading-the-results-honestly) is explicit about why.

## 1. Start the services

If this is a fresh checkout, follow the [installation guide](../ten-minute-install.md)
first: it creates the `.env` this step (and the capture script) reads. Then bring up
postgres, the broker and the PIDSMaker container — postgres first, since it creates the
docker network the other two join:

```bash
docker compose -p postgres -f compose-postgres.yml up -d
docker compose -f compose-kafka.yml up -d
docker compose -f compose-pidsmaker.yml up -d
```

The broker is reachable two ways, which matters because the two halves of this page live
in different places: `kafka:9092` from inside the docker network (the PIDSMaker container)
and `localhost:29092` from the host (SPADE). Both are the defaults everywhere, and the host
port follows `KAFKA_PORT` in `.env` if you change it.

## 2. Build SPADE

Build SPADE from its [`pidsmaker-dev` branch](https://github.com/ashish-gehani/SPADE/tree/pidsmaker-dev)
on the host. It needs **JDK 21 and maven** — and JDK 21 specifically, both to build and to
run. `./configure` will happily accept an older `java` on your `PATH` and the build then
fails deep in maven with `invalid target release: 21`. So install JDK 21 first and point
`JAVA_HOME` at it:

```bash
# Debian/Ubuntu; the package name and path differ on other distros
sudo apt install -y openjdk-21-jdk maven
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version        # must say 21.x before continuing

git clone -b pidsmaker-dev https://github.com/ashish-gehani/SPADE.git ~/SPADE
cd ~/SPADE

# The C audit bridge includes uthash.h, which the repo does not vendor; fetch the
# single header into the source tree before building (or `sudo apt install uthash-dev`).
curl -sSL -o pkg/linux/audit_bridge/src/uthash.h \
    https://raw.githubusercontent.com/troydhanson/uthash/master/src/uthash.h

./configure && make
```

The build produces `lib/spade.jar` and `bin/spadeAuditBridge`. That is all: you do not
need to configure SPADE's Kafka storage or start the server — the capture script does
both, finds the JDK 21 on its own, and takes `--spade-home DIR` if your checkout is not
`~/SPADE`. (No root, or JDK 21 not packaged for your distro? Download a portable build —
e.g. from [Adoptium](https://adoptium.net/temurin/releases/?version=21) — unpack it
anywhere, and pass `--java-home DIR` to the script.)

## 3. Capture a benign period and train on it

The model learns "normal" from what you capture here, so capture a period you believe is
clean. Back in the PIDSMaker checkout, **on the host**, start the capture agent. It records
the host's activity and streams it into the topic `benign` for as long as it runs:

```bash
scripts/capture/new_capture.sh --topic benign
```

It first checks everything it is about to rely on and stops at the first problem with the
fix spelled out — `sudo` for the audit tools and the audit log, auditd running and not
suspended, a SPADE build and a JDK for it, the broker. Then it attaches SPADE to the topic,
wires the audit stream into it, and reports the topic's growth every 30 seconds:

```
==> Checking the setup
  ✓ audit tools installed
  ✓ auditd is running
  ✓ kernel auditing enabled
  ✓ capturing this login session (5781); --user NAME or --system-wide widen it
  ✓ 18G free where auditd writes
  ✓ SPADE build found in /home/you/SPADE
  ✓ JDK 21 found for SPADE
  ✓ SPADE server is running
  ✓ Kafka broker is up
  ✓ broker reachable from the host at localhost:29092
  ✓ all checks passed

==> Starting the capture
  ✓ topic 'benign' created
  ✓ SPADE publishes to 'benign' via localhost:29092
  ✓ SPADE is reading the audit stream
  ✓ audit rules armed (key pidsmaker_20260910_071533)

==> Capturing to topic 'benign' - Ctrl-C to stop
    Read it live from another terminal - score it with a trained model:
      docker compose -f compose-pidsmaker.yml exec pids \
        python pidsmaker/stream_detect.py orthrus SPADE_AUDIT --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml \
        --stream_topic=benign --stream_from_beginning=False --emit_viz=True
    or turn it into a dataset to train on (reads the whole topic, stops when it is quiet):
      docker compose -f compose-pidsmaker.yml exec pids \
        python pidsmaker/stream_ingest.py SPADE_AUDIT --stream_topic=benign --stream_idle_timeout=25

    07:15:51  topic 'benign': 0 records (+0 this capture, 0 min)
    07:16:21  topic 'benign': 0 records (+0 this capture, 0 min)
    07:16:50  topic 'benign': 10,602 records (+10,602 this capture, 1 min)
    07:17:21  topic 'benign': 20,393 records (+20,393 this capture, 1 min)
```

**Now generate the activity, in a second terminal.** The recording follows your login
session, so use the machine as you normally would. To have something reproducible, the
repository ships a scripted benign workload — shells forking tools that read files, write
output, resolve names, fetch a page, archive a directory:

```bash
bash scripts/workload/benign.sh /tmp/pidsmaker_workload 45     # about three minutes
```

Watch the record count climb in the first terminal, then press **Ctrl-C** there to stop.
The script removes its audit rules, closes the stream, waits for SPADE to publish what it
still holds, and detaches:

```
==> Stopping
  ✓ audit rules removed
    Waiting for SPADE to publish what it has...
  ✓ topic 'benign' holds 29,061 records (+29,061 from this capture)

==> Done
    SPADE keeps running; run this again with --topic benign to keep adding to the topic.
```

!!! warning "Capture from a quiet session"
    With read/write auditing on, an IDE such as VS Code (its server process, extension
    hosts, file watchers) or a browser produces tens of thousands of audit records per
    second by itself, which swamps the capture and fills auditd's log at several MB/s.
    Capture from a login session that does not contain them — a plain SSH login is
    simplest — or drop them by name with `--ignore node,code`. The count line in the
    capture terminal tells you at once: a few hundred records per second is a workstation
    being used; tens of thousands is an IDE.

!!! note "Why the count jumps at the end, and lags while capturing"
    SPADE's audit bridge keeps the newest **10,000 audit events** in a reordering buffer
    so it can emit them in order, and publishes an event only once 10,000 newer ones have
    arrived. On a busy host that is a lag of seconds; on a quiet session it can be
    minutes, and the last few thousand events of a capture only reach the topic when you
    stop it (the "Waiting for SPADE to publish what it has" step). A detector watching the
    topic sees the same lag. It is inherent to SPADE's input path, not to the script.

**Build the dataset**, inside the container. This is the second command the capture
printed: it reads the whole topic into the postgres schema the offline pipeline uses,
stops once the topic has been quiet for 25 seconds, and writes the dataset config.

The dataset name is yours to choose. `stream_ingest.py MYHOST …` creates the postgres
database `myhost` and writes `/home/artifacts/streaming/myhost/dataset.yml`; every later
command then takes `MYHOST` with that file, and its artifacts live apart from any other
dataset's. The file records which built-in dataset the new one inherits its conventions
from (edge vocabulary, node types) — `SPADE_AUDIT` for anything SPADE produces, or
`--template` to say otherwise. This page uses `SPADE_AUDIT` itself as the name, for brevity:

```bash
docker compose -f compose-pidsmaker.yml exec pids \
    python pidsmaker/stream_ingest.py SPADE_AUDIT --stream_topic=benign --stream_idle_timeout=25
```

```
Kafka source: kafka:9092 topics=['benign'] group=pidsmaker-1788979485-2071 (from the beginning)
Ingested 595547 events and 7322 nodes.
Dataset config written to /home/artifacts/streaming/spade_audit/dataset.yml
  dataset: SPADE_AUDIT (database `spade_audit`, conventions of SPADE_AUDIT)
Warning: the capture spans a single day, which cannot be split into train/val/test.
Next step - train a system on this capture:
  python pidsmaker/main.py orthrus SPADE_AUDIT --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml --save_model
```

The database now holds the capture in the exact schema a pre-processed DARPA dataset
has, and `dataset.yml` holds the dates it covers and their split. A dataset is split by
calendar day, so **a capture spanning several days gets a real train/val/test split**;
with fewer, ingestion warns and validates on the training day (step 6 explains what that
costs). The ingest also saved its node knowledge (`stream_state.pkl`), which a detector
joining the topic later uses to resolve edges whose vertices were published before it
started.

**Train on it.** This is the normal PIDSMaker pipeline — featurization, batching,
training — pointed at that dataset, with one addition:

```bash
docker compose -f compose-pidsmaker.yml exec pids \
    python pidsmaker/main.py orthrus SPADE_AUDIT --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml --save_model
```

```
[@epoch03] Test finished - Test Loss: 1.3291
[@epoch07] Test finished - Test Loss: 1.3062
[@epoch11] Test finished - Test Loss: 1.2840
Model weights saved to /home/artifacts/training/training/c5f9609e…/SPADE_AUDIT/trained_models/model_best
Dataset SPADE_AUDIT declares no ground truth: skipping evaluation. Run `pidsmaker/stream_detect.py` to score a live stream with the trained model.
```

`--save_model` is what makes the detector possible: it writes the model's weights to disk
next to the run's other artifacts. It is off by default because the weights of a
memory-based encoder (TGN) scale with the training graph, and keeping them for every
experiment would pile up.
Evaluation is skipped on purpose: a benign capture has no labelled attacks to score
against. What training does leave behind, and what the detector needs, is the model and
the distribution of validation losses its threshold comes from.

*(These ingest and training lines are from the longer capture used throughout this page —
about 1.7 million audit records; a three-minute workload gives the same lines with
smaller counts.)*

You now have:

| What | Where | Why it matters |
|---|---|---|
| The topic | `benign` on the broker | What every PIDSMaker command reads, by name |
| The dataset | postgres database `spade_audit` + `/home/artifacts/streaming/spade_audit/dataset.yml` (in the container) | The exact schema the offline pipeline reads; the config gives the dates found in the capture and their split |
| The model | `/home/artifacts/training/.../model_best` | Written by `--save_model`; what the detector scores with, plus the validation losses its threshold comes from |

!!! note "How the detector finds the model"
    You never name the model. PIDSMaker derives each task's artifact directory from a
    hash of the arguments that affect it, and `stream_detect.py` computes the same hash
    from the same arguments — so giving the detector the **same system, dataset name and
    `--dataset_config`** as the training command selects the model that command produced.
    Changing any of them (or re-ingesting the dataset, which rewrites its dates) points at
    a different, not-yet-trained run; `--model_checkpoint DIR` overrides the lookup.

### Choosing what to capture

By default the agent records **your login session** — every process descended from the
shell you started it in (tmux panes and VS Code terminals of the same window belong to it;
a new SSH login does not). Two options widen that, and `--fresh` empties the topic first:

| Option | Records |
|---|---|
| *(default)* | Every process of this login session |
| `--user NAME` | Every process of that user, in all of their sessions |
| `--system-wide` | Every process on the machine — what a deployment records |
| `--ignore node,code` | Drops processes by name — for an IDE, browser or agent whose constant file I/O would swamp everything else |
| `--fresh` | Empty the topic before capturing |

Run the agent again with the same `--topic` and the new capture is **added** to it; the
script says so, with the count already there. So a representative dataset is several
captures into one topic — say, a working day at a time with `--system-wide` — and an
experiment is a new topic name. `scripts/capture/new_capture.sh --help` lists every option.

## 4. Detect in real time

This is the deployment shape: the capture agent runs, the detector watches the topic, and
whatever happens on the host is scored as its time windows close. Three terminals.

**Terminal 1 — capture** into a new topic:

```bash
scripts/capture/new_capture.sh --topic incident
```

**Terminal 2 — the detector**, started now so it sees what comes next.
`--stream_from_beginning=False` means "only what is published from now on"; with no idle
timeout it runs until you stop it; `--emit_viz` records every score for step 5:

```bash
docker compose -f compose-pidsmaker.yml exec pids \
    python pidsmaker/stream_detect.py orthrus SPADE_AUDIT \
        --dataset_config=/home/artifacts/streaming/spade_audit/dataset.yml \
        --stream_topic=incident --stream_from_beginning=False --emit_viz=True \
        --stream_window_size=1
```

Nothing prints until a window closes. The model's windows are 15 minutes, so on a
continuously busy host the first line appears 15 minutes after the first record; add
`--stream_window_size=1` while you watch live to close a window every minute instead
(the model was trained on 15-minute windows, so scores differ slightly from a
same-sized window — fine for watching, not for measuring).

**Terminal 3 — the activity**: the benign workload again, then the staged intrusion. It
drops a payload into a hidden directory, executes it, reads a decoy credentials file and
sends it to a listener on the loopback interface, then writes a persistence file. No real
secret, nothing leaves the machine:

```bash
bash scripts/workload/benign.sh   /tmp/pidsmaker_workload 45
bash scripts/workload/incident.sh /tmp/pidsmaker_workload
```

Back in terminal 2, the detector reports each window as it closes. A window closes when
the stream moves past its end (15 minutes by default) or nothing has arrived for
`--stream_flush_timeout` seconds, so a quiet host still gets scored:

```
Kafka source: kafka:9092 topics=['incident'] group=pidsmaker-1789049768-3511 (new records only)
Restored 7902 nodes and 164917 vertex ids from the stream state: edges whose vertices
were published before this run started can still be resolved.
Detection threshold: 12.0464 (method: max_val_loss)
Window 15 min | flush 60s idle | threshold 12.0464 (max_val_loss)
Detector ready, waiting for provenance...
Kafka partitions assigned: incident[0]
[2026-09-09 15:10:59~2026-09-09 15:11:00] nodes=49 edges=91 max_score=6.9841 thr=12.0464 alerts=0 (1897 ms)
[2026-09-09 15:11:00~2026-09-09 15:11:02] nodes=65 edges=124 max_score=10.6315 thr=12.0464 alerts=0 (15 ms)
```

Each line is one window: how many nodes and edges it held, the highest anomaly score in
it against the threshold, and how many nodes crossed it. Any node above the threshold is
printed under its window line, in the form:

```
  ALERT node=1735 [subject] score=13.4400 subject /usr/bin/curl curl https://example.com
```

With `--alert_sink=stdout,file --alert_file=...` each alert is also one JSON object per
line (`label`, `node`, `node_type`, `score`, `threshold`, `window`, and `source_key`, the
SPADE vertex hash the node came from). When you are done, **Ctrl-C the capture** in
terminal 1 (it flushes what SPADE still holds), then **Ctrl-C the detector**: it scores the
window in flight, saves its node knowledge, and builds the viewer.

Whether those two windows raise an alert depends on the model, which is the subject of
step 6; what this step proves is that provenance flows from a syscall to a scored node
while the machine is being used.

## 5. Explore the predictions

The alert lines only carry what crossed the threshold. With `--emit_viz` the detector also
recorded *every* scored node, in two forms under a run directory the web viewer discovers
on its own (`<ARTIFACTS_DIR>/detection/evaluation/stream_<timestamp>/SPADE_AUDIT/`):

- **`stream_scores.jsonl`** — one line per node per window
  (`window`, `node`, `node_type`, `label`, `score`, `is_alert`, `source_key`). This is the
  flat record for pandas/SQL: load it and plot the score distribution, rank nodes, or track
  one node across windows.
- **The web viewer's artifacts** — every node as a 3D point per time window, coloured by
  anomaly score. Open them with the same viewer the offline pipeline uses:

```bash
docker compose -f compose-pidsmaker.yml exec pids python -m pidsmaker.vizgen.web.viz_server
```

Open the printed URL; the run appears in the browser's run list. Pick it to scrub through
the windows, click a node to see its path, type and score, and follow its edges. Two buttons
in the side panel are the quickest way in:

- **Top Malicious Events** lists the scored edges from the highest score down — source and
  destination paths, edge type, time window — with *Load more* to page through all of them
  and a CSV export. On an incident capture, the hidden `updater` payload and the decoy
  files it reads should be near the top.
- **Plot Score Distribution** shows the shape of "normal" and where the threshold cuts it.

(The run shows as a *word2vec* / base-embedding run: the detector visualizes the
featurization space, not the GNN encoder.) The dimensionality reduction runs when the
detector stops, so `--emit_viz` adds a little time and memory there;
`--viz_max_points` caps how many node-snapshots are kept.

## 6. Reading the results honestly

Whether step 4 raised alerts or not, treat the number with care. Two reasons, both about
the shape of the captures rather than the pipeline:

- **A few minutes is one or two time windows.** A short capture is well under the window
  size, so there is little temporal spread for the detector to work across.
- **train = val = test.** With a single day of benign activity, the ingest warned it could
  not split, so the threshold (`max_val_loss`) is the largest loss the model saw *on the
  very data it trained on*. Only activity that looks genuinely different from that
  capture can exceed it. This is deliberately conservative: it guarantees no false
  positives on the training window, at the cost of sensitivity.

To get a result that means something about detection you need a benign capture with real
structure over time:

- **Capture across several real days**, into one topic —
  `scripts/capture/new_capture.sh --topic benign --system-wide` for a working day, each day.
  Ingestion then finds several dates and makes a real train/val/test split, so the
  threshold is computed on held-out benign activity.
- **Then run the detector on a later day**, live as in step 4, and introduce the incident.
  Now the intrusion is being judged against a "normal" the model actually learned.
- **The threshold is the knob that matters most**, and it is a plain argument of the
  detector command:

```bash
# default: largest validation loss — nothing benign in validation would alert
--evaluation.node_evaluation.threshold_method=max_val_loss

# more sensitive, far noisier
--evaluation.node_evaluation.threshold_method=mean_val_loss

# or pin it yourself, e.g. from a percentile of a known-quiet day
--alert_threshold=8.5
```

The `max_score` and `thr` on each window line tell you how much headroom a quiet period
has, which is what you calibrate against.

## 7. Going further

**Another system.** The topic is a dataset like any other, so swap `orthrus` for `flash`,
`magic`, `threatrace`, `kairos`, or anything in the [systems list](../tuned_systems.md) in
both the training and the detector command. Node-level systems score nodes directly
rather than through their edges, which changes what an alert means. Two caveats are
listed under [Limitations](#limitations).

**Score a capture you already have.** `--stream_from_beginning=True --stream_idle_timeout=25`
makes the detector read a whole topic and exit when it is drained — every run reads under
a fresh Kafka consumer group, so this always means the *whole* topic.

**Publish an audit log instead of capturing.** Given `ausearch --raw` output from another
machine or an earlier day, `scripts/capture/new_capture.sh --topic NAME --from-log FILE`
runs it through the same SPADE reporter and publishes it to the topic.

**Replay a file instead of a topic.** Point SPADE's Kafka storage at a file
(`add storage Kafka kafka.output.file=/path/replay.json`) and feed the result straight
through the detector with `--stream_source=file --stream_file=... --stream_format=avro_json`
— useful for testing a threshold change against a period you have already seen.

**Alerts somewhere else.** `--alert_sink=kafka --alert_topic=pids-alerts` publishes them for
a SIEM to pick up; the sinks compose (`--alert_sink=stdout,file,kafka`). See
[Where alerts go](#where-alerts-go).

---

## How it works

### The path of a record

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

Inside PIDSMaker (`pidsmaker/streaming/`), a record goes through three stages that are
deliberately separate, because each answers a different question:

| Stage | Question it answers | Swap it with |
|---|---|---|
| **Source** | Where do payloads come from? | `--stream_source=kafka` / `file` |
| **Decoder** | What encoding are they in? | `--stream_format=avro` / `avro_json` / `json` |
| **Adapter** | What do they *mean*? | `--stream_adapter=spade` |

Only the adapter knows a producer's schema. It emits `StreamNode` and `StreamEvent`
records (`pidsmaker/streaming/records.py`), and everything downstream — windowing,
featurization, batching, detection, the postgres sink — is producer-agnostic. SPADE's
schema is deliberately simple — a stream of vertices and edges carrying annotations — which
is why it is the right place to branch onto.

### How a live window is scored

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

!!! note "One unavoidable difference"
    Offline, a node's score is its maximum over the whole test set. A detector cannot know
    the future, so a node is scored within each window, as soon as that window closes.

Which featurizers can be served online follows from step 2:

| Method | Online | Why |
|---|---|---|
| `word2vec`, `fasttext`, `doc2vec`, `hierarchical_hashing` | ✅ | Embed a node from its own label |
| `only_type`, `only_ones` | ✅ | No embedding at all |
| `flash` | ✅ | Embeds from the node's own events, accumulated per window |
| `temporal_rw`, `alacarte`, `ocrapt_features` | ❌ | Need random walks or dataset-wide statistics, which a live stream cannot provide |

A system trained with one of the last three raises a clear error rather than silently
producing wrong vectors.

### Joining a stream in progress

SPADE publishes each vertex **once** (its default `Deduplicate` screen), and every later
edge refers to it by hash alone. A detector started after the capture — the normal case,
since training happens in between — would therefore receive edges whose endpoints it has
never seen and have to drop them. `stream_detect.py` avoids that by loading
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

### What the SPADE adapter does

Two translations, both worth knowing about when reading its output:

**Direction.** A SPADE edge points from the *effect* to the *cause*
(`Used(child=process, parent=file)` for a read). PIDSMaker orients edges along the
information flow, so `src` is always the parent and `dst` the child — which is exactly the
DARPA convention of `file → subject` for a read.

**Vocabulary.** SPADE reports the full system call surface, sometimes qualified
(`"mmap (read)"`). Those operations are mapped onto the 28 edge types of `rel2id_spade`
(`pidsmaker/utils/dataset_utils.py`); unmapped ones are dropped unless
`--stream_keep_unmapped_operations=True`.

Two behaviours are configurable because they are judgement calls:

- `--stream_merge_artifact_versions` (default on) treats every version of a file as one
  node. SPADE's `versions=true` emits a fresh vertex per write, which would otherwise turn
  a single file into hundreds of nodes sharing no history.
- `--stream_keep_agents` (default off) keeps edges to `Agent` (user/group) vertices, which
  have no PIDSMaker node type.

### What the capture agent does

`scripts/capture/new_capture.sh` wires three things together, after checking every
precondition:

1. **Audit rules**, keyed so they can be found and removed again — SPADE's own syscall
   set with file and network I/O, scoped by `-F sessionid=` (default), `-F auid=`
   (`--user`) or nothing (`--system-wide`). The agent's own processes and SPADE's are
   excluded by pid, or SPADE reading its input would be recorded and fed back to itself.
2. **A live feed**: `tail -F` on auditd's log (as root), filtered to this run's key by
   audit serial, written into a **named pipe**.
3. **SPADE's Audit reporter reading that pipe** as its `inputLog=` — a pipe blocks SPADE
   until data arrives, so the same reporter that replays a file becomes a live one, and
   closing the pipe on Ctrl-C ends it cleanly. Its Kafka storage takes the broker and
   topic as arguments, so nothing in SPADE's `cfg/` is edited.

SPADE stays running when the agent exits, with its storage still attached. For reference,
these are the SPADE commands the agent issues:

```sh
bin/spade start
printf 'add storage Kafka kafka.output.server=localhost:29092 kafka.output.topic=benign\nexit\n' | bin/spade control
printf 'add reporter Audit inputLog=/path/to/pipe fileIO=true netIO=true\nexit\n' | bin/spade control
```

`fileIO=true netIO=true` matter: SPADE's defaults suppress `read`/`write` and
`send`/`recv` edges — the bulk of the data flow a PIDS learns from. SPADE also has a native
live mode (`add reporter Audit` with no `inputLog`), which reads the audit dispatcher's
socket directly; it needs SPADE running as root and the `af_unix` audisp plugin enabled
host-wide, which is why the agent uses the pipe.

## Arguments

Streaming is not one of the pipeline tasks — it feeds them (ingestion) or consumes their
output (detection) — so its settings are plain `--stream_*` arguments that never enter
the task hashes. Both `stream_ingest.py` and `stream_detect.py` take them:

| Argument | Default | Meaning |
|---|---|---|
| `--stream_topic` | *required* | The topic the capture agent publishes to (`--topic NAME`); several can be given, comma-separated |
| `--stream_brokers` | `kafka:9092` | Bootstrap servers, as seen from the container |
| `--stream_from_beginning` | `True` | Read the whole topic (`True`) or only records published from now on (`False`) |
| `--stream_group_id` | a new group per run | Consumer group; name one to resume from its last committed offset |
| `--stream_idle_timeout` | `0` | Stop after N seconds without a record (0 = run until interrupted) |
| `--stream_max_records` | `0` | Stop after N records |
| `--stream_window_size` | `construction.time_window_size` | Window size in minutes |
| `--stream_flush_timeout` | `60` | Close a pending window after N seconds of wall clock |
| `--stream_source` | `kafka` | `kafka`, or `file` to replay a capture file |
| `--stream_file`, `--stream_follow`, `--stream_rate` | — | The capture file, whether to tail it, and a replay speed in records/s |
| `--stream_adapter` | `spade` | Which producer's schema the stream carries |
| `--stream_format` | adapter's | `avro`, `avro_json`, `json`, `avro_container` |
| `--stream_schema` | adapter's | `.avsc` path, or a name bundled in `pidsmaker/streaming/schemas` |
| `--stream_on_error` | `skip` | `skip` or `raise` on an unreadable record |
| `--stream_merge_artifact_versions`, `--stream_keep_unmapped_operations`, `--stream_keep_agents` | `True`, `False`, `False` | SPADE adapter behaviours, see above |

`stream_detect.py` adds:

| Argument | Default | Meaning |
|---|---|---|
| `--alert_sink` | `stdout` | Where detections go, comma-separated: `stdout`, `file`, `kafka`, `none` |
| `--alert_file`, `--alert_topic` | — | Destination of the `file` and `kafka` sinks |
| `--alert_log_all_windows` | `False` | Also record windows with no alert in the file sink |
| `--alert_threshold` | from the trained run | Detection threshold, overriding the one derived from the validation losses |
| `--model_checkpoint` | the run's `model_best` | Trained model directory |
| `--stream_state` | `<artifacts>/streaming/<database>/stream_state.pkl` | Node knowledge to load; `none` to start blank |
| `--stream_state_save` | `True` | Write the updated node knowledge back on exit |
| `--stream_max_nodes`, `--stream_max_events` | `2000000` | Node capacity of the batching tables; how much history the TGN neighbor loader keeps before its context is reset |
| `--stream_stats_every` | `20` | Log throughput every N windows (0 = never) |
| `--emit_viz`, `--viz_method`, `--viz_max_points` | `False`, `umap`, `300000` | Record every scored node for the web viewer; reduction method; cap on retained snapshots |

`stream_ingest.py` adds:

| Argument | Default | Meaning |
|---|---|---|
| `--template` | matches the adapter (`SPADE_AUDIT`) | Built-in dataset whose conventions the new one inherits |
| `--append` | `False` | Add to the existing dataset instead of replacing it |
| `--val_ratio`, `--test_ratio` | `0.15` | Share of the captured *days* used for validation and test |
| `--dataset_config_out`, `--stream_state_out` | under `<artifacts>/streaming/<database>/` | Where `dataset.yml` and `stream_state.pkl` are written |
| `--database_host`, `--database_user`, `--database_password`, `--database_port` | `postgres`, `postgres`, `postgres`, `5432` | Postgres connection |

And two arguments of the regular pipeline (`main.py`) exist for streamed datasets:

| Argument | Meaning |
|---|---|
| `--dataset_config FILE` | The `dataset.yml` the ingest wrote: name, database, template and dates of a dataset that is not built in |
| `--save_model` | Write the trained weights to `training/<hash>/trained_models/model_best`, which the detector loads. Off by default |

### Where alerts go

`--alert_sink` takes a comma-separated list:

| Sink | Output |
|---|---|
| `stdout` | A line per window and per alert |
| `file` | One JSON object per alert (`--alert_file`), appended |
| `kafka` | The same JSON published to `--alert_topic`, for a SIEM to pick up |
| `none` | Discards everything, for throughput measurements |

## Adding another provenance source

Everything except the adapter is already source-agnostic. To branch a new producer:

1. Subclass `ProvenanceAdapter` in `pidsmaker/streaming/adapters/`, implementing
   `handle(record)` so it yields `StreamNode` / `StreamEvent` objects. Two things are
   yours to get right: mapping the producer's entity types onto `subject`/`file`/`netflow`,
   and orienting edges along the **information flow** (`src` → `dst`: a read is
   `file → subject`, a write is `subject → file`).
2. Register it in `ADAPTERS` (`pidsmaker/streaming/adapters/__init__.py`).
3. If its vocabulary differs, add a `rel2id_*` to `pidsmaker/utils/dataset_utils.py` and a
   dataset entry in `pidsmaker/config/config.py` declaring `num_edge_types`, to serve as
   the template of the datasets it produces.

Then `--stream_adapter=<name>` is all that changes; the source, decoder, ingestion,
detection and alerting stay as they are. If the producer also publishes to Kafka with Avro,
nothing else needs writing at all.

## Limitations

- **Detection quality is a property of the trained model**, not of the transport. A system
  trained on a short or unrepresentative capture will alert accordingly.
- **`magic`'s threshold** is computed from the test set's own embedding distances, which a
  live stream has no equivalent of. Pass `--alert_threshold` to run it.
- **Capacity is bounded**: `--stream_max_nodes` sizes the batching tables, and reaching
  `--stream_max_events` resets the TGN temporal context rather than growing without bound.
- **A single stream is a single host.** Watching several hosts means one detector per
  topic, or an adapter that partitions by host.
- **Latency is SPADE's**: provenance reaches the topic about 10,000 audit events after it
  happened (see the note in [step 3](#3-capture-a-benign-period-and-train-on-it)).

## Troubleshooting

The agent's checks catch most problems before anything runs; this is for what they cannot see.

| Symptom | Cause |
|---|---|
| A check fails | Its line says what to do; `--check` re-runs only the checks. `--help` lists the options that relocate things (`--spade-home`, `--java-home`, `--kafka-server`, `--audit-log`). |
| `cannot read /var/log/audit/audit.log as root without a password prompt` | The agent needs `sudo` without a prompt once it is running. Run `sudo -v` first (or run the script with `sudo`). |
| `auditd has SUSPENDED logging` | auditd stopped writing when its disk ran low and stays that way, even after space is freed and even though it shows as active. Free space, then `sudo systemctl restart auditd`. |
| The count grows by tens of thousands per second | An IDE or browser in the captured session. Capture from a plain SSH login, or `--ignore node,code`. Stop the capture before auditd's disk fills. |
| The count stays at 0 for a while | SPADE's reordering buffer (10,000 events, see step 3), or nothing is happening in the captured scope — a session capture records only this login session. |
| `the feed stopped (SPADE closed the pipe?)` | SPADE's reporter died; `$SPADE_HOME/log/current.log` says why. Run the agent again. |
| Far fewer events than syscalls run | SPADE dropped records it could not parse; its `log/current.log` has the exception. |
| SPADE `make` fails with `invalid target release: 21` | maven is running under a JDK older than 21. `export JAVA_HOME` to a JDK 21 (and put its `bin` on `PATH`) before `make` — see [step 2](#2-build-spade). |
| SPADE `make` fails with `uthash.h: No such file or directory` | The audit bridge's header is not vendored. Fetch it into `pkg/linux/audit_bridge/src/` (or `apt install uthash-dev`) — see [step 2](#2-build-spade). |
| `control port 19999 is held by pid N, not this build's kernel` / `SSLHandshakeException` | A stray SPADE kernel (often an older build with its own keys) holds the control port. The message names the pid to `kill`; then run the agent again. |
| `its control client did not answer` | The kernel is up but its control channel is wedged. `bin/spade stop && bin/spade start` in the SPADE checkout. |
| The detector reads nothing from a topic that has records | It was started with `--stream_from_beginning=False` after the records were published: it only sees what comes next. Use `True` to read the whole topic. |
| `No trained model at .../model_best` | Training ran without `--save_model`, or with different arguments (or the dataset was re-ingested since, which changes its dates and therefore the run). Train again with `--save_model` and the exact arguments the detector uses. |
| `database "postgres" has a collation version mismatch` | A postgres warning unrelated to PIDSMaker; the ingest works around it. |
| 0 alerts / alerts on everything | Expected on a single short or unrepresentative capture. See [step 6](#6-reading-the-results-honestly). |
