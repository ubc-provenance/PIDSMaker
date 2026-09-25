# PIDSMaker Web Visualizer

An interactive 3D viewer for PIDSMaker embeddings. A Flask server reads
pre-computed artifacts and serves a Three.js front end in the browser. It is
independent of the pipeline: you generate the data for a run and select it in the
browser, rather than viewing being a pipeline step.

## Run the viewer

Start the server inside the Docker container (Flask and numpy are installed by
the Dockerfile):

```bash
python -m pidsmaker.vizgen.web.viz_server --host 0.0.0.0 --port 5000
```

Options: `--host` (default `127.0.0.1`; use `0.0.0.0` to reach it from outside),
`--port` (default `5000`), `--debug`. The artifacts root is `/home/artifacts` if
present, otherwise `$PIDS_ARTIFACTS_DIR`, otherwise `<repo>/artifacts`.

## Artifacts location

`compose-pidsmaker.yml` mounts the host path `${ARTIFACTS_DIR:-/home/artifacts}`
at the container's artifacts root `/home/artifacts`. If your `.env` sets
`ARTIFACTS_DIR` (for example `ARTIFACTS_DIR=./artifacts`), that folder is what
gets served — point it at the directory that actually holds your runs.

**Run Browser empty?** The viewer only discovers runs under
`evaluation/evaluation/<hash>/<dataset>` (and `detection/evaluation/...`) that
have a `viz_manifest.json` and/or `viz/*_points.json`. If it shows nothing,
check that `ARTIFACTS_DIR` points at the folder containing those runs, then
recreate the container (e.g. `ARTIFACTS_DIR=/home/artifacts docker compose
-f compose-pidsmaker.yml up -d`) and restart the server. A standalone top-level
`<root>/viz/` export is not discovered — it must live inside a run directory.

## View it

The standard way to view is an SSH tunnel to `localhost`. It is secure, exposes
nothing, and works from any machine that can reach the server:

```bash
ssh -N -L 5000:localhost:5000 <user>@<server>
# then open http://localhost:5000
```

If the server runs in Docker and the port is not published to the host, tunnel to
the container IP: `-L 5000:<container-ip>:5000`. On the server itself, open
`http://localhost:5000` directly. `compose-pidsmaker.yml` also maps `5000:5000`
for optional LAN or VPN access; keep it firewalled, as there is no
authentication.

The last run you viewed loads automatically. Use **Open Run Browser** (top of the
left panel) to pick another.

Rendering happens on the GPU of the machine that opens the browser. A workstation
GPU renders the full point cloud smoothly; a laptop may stutter, and low-memory
machines can lag when the adjacency data is loaded. See `ARCHITECTURE.md`,
section 10.

## Generating viz data

The viewer reads `embedding_viz_*_points.json` and `*_adj.json`. Evaluated runs
without them appear in the Run Browser badged `Needs viz` or `Partial`; `Ready`
runs can be regenerated with the regenerate button.

### Opt in when training: `--save_for_viz`

Generation is fastest and most complete when the training run **pre-saved** the
extra artifacts the exporter reuses: the per-epoch model checkpoints (for the
encoder embedding space and the epoch selector) and a cached copy of the test
graphs. Saving those on **every** run does not scale — the checkpoints and cache
are large and most runs never open the viewer — so they are **off by default** and
enabled per run with a flag:

```bash
python pidsmaker/main.py <model> <dataset> --save_for_viz
```

Enable it on the runs you actually intend to explore. Without it, a run is still
discoverable and you can still generate viz data for it, but:

- the **featurization** embedding space works normally (it does not need the
  checkpoints);
- the **GNN encoder** space needs the per-epoch checkpoints — if they were not
  saved, encoder generation is skipped with a warning;
- the test graphs are **recomputed** on demand (the whole batching pipeline reruns),
  which is slower but produces identical data.

So `--save_for_viz` trades disk up front for fast, full generation later; leaving
it off keeps runs lean and defers (or limits) the cost to generation time.

### First, make existing runs visible (backfill manifests)

A run only shows up in the Run Browser once it has a `viz_manifest.json` — a small
index file pointing at the run's trained model, graphs, and per-epoch scores (it
holds no embeddings; it just tells the viewer where things already are). New
evaluations write it automatically, but runs evaluated **before this feature** (or
copied in from elsewhere) have none, so the Run Browser shows nothing even though
the runs sit in your artifacts folder.

If you cloned the repo and already have evaluated runs, generate the manifests
once — it only reads files already on disk; nothing is retrained:

```bash
# inside the container (where the artifacts folder is mounted)
python -m scripts.backfill_viz_manifests --dry-run   # preview what it will write
python -m scripts.backfill_viz_manifests             # write the manifests
```

It walks `evaluation/evaluation/*/*` and `detection/evaluation/*/*`, skips runs
that already have a manifest (use `--force` to rewrite) and incomplete runs with
no `precision_recall_dir`, and uses the same artifacts root as the server
(override with `--artifacts-root`). Refresh the viewer afterwards: the runs appear
badged `Needs viz`, ready to generate. The manifest records paths under
`training/` and `batching/`, so generation also needs those artifacts present, not
just `evaluation/`.

### From the browser (recommended)

Open the Run Browser, click **Generate** on a run, choose the options (Word2Vec,
GNN Encoder, or both; all epochs; UMAP or t-SNE; sampling), then click
**Start generation**. A live console streams the generation logs with a phase bar
(Start, Load graphs, Model, Embeddings, UMAP, Write). When it finishes, click
**Visualize now**. The job runs on the server; closing the console leaves a
status pill (bottom-right) that survives a page refresh and reopens the console.
One generation runs at a time.

Generation executes model-loading code, so it is gated by the environment
variable `PIDS_VIZ_ALLOW_EXPORT` (on by default). Set it to `0` or `false` to
disable the `/api/export*` endpoints on a shared or exposed deployment, and keep
the server bound to `127.0.0.1`.

### From the command line

The same generator without a browser. It loads the run's trained model and
graphs, computes embeddings, and runs UMAP:

```bash
# latest evaluated run for the dataset
python -m pidsmaker.vizgen.web.export <model> <dataset> --embeddings both

# a specific run from the artifacts folder
python -m pidsmaker.vizgen.web.export <model> <dataset> \
    --run /home/artifacts/evaluation/evaluation/<hash>/<dataset>
```

`scripts/embedding_viz.py` is a thin launcher for the same command. After it
finishes, refresh the viewer and select the run.

### Large datasets (E5) — sampling caps

`--max_benign` / `--max_attack` (and the **Max benign** / **Max attack** fields
in the generate dialog) default to `all`, i.e. no cap. Export time is dominated
by the dimensionality reduction. The preferred GPU path (RAPIDS cuML UMAP)
uses approximate nearest-neighbour search and scales well past a million
nodes, but the CPU-fallback hybrid path precomputes an exact k-NN with a
batched GPU `cdist` (`dimensionality_reduction.py`), which is O(n²) in the
node count and is the real bottleneck when it is the path in use; sklearn's
t-SNE runs Barnes-Hut by default, not the exact O(n²) method, so it scales
better than a naive implementation but is still far slower than UMAP at this
scale. The `*_points.json` / `*_adj.json` files also grow linearly with kept
nodes/edges, so for E5-scale runs (millions of nodes) leave the caps at `all`
at your own risk; set finite values (e.g. `--max_benign 200000 --max_attack
all`) to keep generation and file sizes tractable. Attack nodes are few, so
capping benign nodes is usually enough; benign sampling is neighbourhood-aware
and keeps benign
nodes adjacent to attack nodes.

The **viewer** also scales independently of generation. Opening a run streams
only the **binary point buffer** in `LOD_CHUNK`-sized steps that each render on
arrival, so the cloud paints as it loads. The nodes are kept in the exporter's
**natural order** — no reordering. Occlusion is resolved by true 3D depth
(`depthWrite:true`), not draw order, so malicious (red) points in front of the
cluster stay visible and ones behind are correctly hidden. Everything the cloud
needs to render, filter (label/detection/type), colour, size and animate over
time is in the buffer.

The buffer is packed to minimise transfer (the SSH-tunnel byte count is the real
cost at scale, ~64 B/node → **~19 B/node**):

- **positions** as `float16` (UMAP coordinates need nowhere near `float32`);
- **no colour column** — every colour is derived on the client from the flags
  (`nodeRgba`), so shipping colours was ~25 % of the buffer wasted;
- **flags** packed to `node_id` (`uint32`) + one `meta` byte (`label<<0 |
  det<<1 | type<<3`);
- **attrs** packed to `uint16[4]` — `tw_idx/tw_start/tw_end` as `uint16` and the
  score as `float16`; `size` is dropped (derived from the label).

Loading is **tiered** so the cloud appears as early as possible (each column is a
separate Range request, so tiers cost nothing extra to split):

1. **Tier 1 — display:** positions + the `meta` byte (colour + size derive from
   it). As soon as these stream in the cloud renders — the *only* thing the
   loader waits on. ~37 % of the buffer.
2. **Tier 2 — interact/temporal:** the packed `attrs` + node `ids`, fetched in the
   **background** once the cloud is on screen; the time slider, playback, heatmap,
   false-positive and attack-graph overlays, and the id→instances map enable when
   it lands (`S.auxReady`).
3. **Tier 3 — on demand:** per-node string metadata (`/api/node`, on selection)
   and edges (`/api/neighbors`, `/api/causal`, `/api/campaign`) — fetched only
   when the user asks, never up front.

Chunks **grow** (`LOD_CHUNK` → doubling to `LOD_CHUNK_MAX`) so a big run streams
in ~12 round trips instead of ~80, while the first (small) chunk still paints fast.

The **string metadata** (paths, command lines, type names, …) is **never
bulk-loaded**, and neither is the node-id column up front. Selection is keyed on
the **buffer row index**: clicking a point fetches just that row's (and its 20 KNN
neighbours') metadata via `GET /api/node?idxs=…` — an O(1) store lookup, instant
even at 20 M. Text search and the CSV filter are resolved server-side
(`GET /api/search`, `POST /api/filter`) by scanning the raw path/cmd blobs with
one regex (no per-row decode) and returning matching **rows** (== buffer indices),
so the client filters by index without the ids. Opening a run transfers **zero
metadata**. Beyond `LOD_CAP` resident nodes the view keeps the first `LOD_CAP`
(the browser/GPU is the limit, not the server); the constants are `LOD_CAP` /
`LOD_CHUNK` / `LOD_CHUNK_MAX` in `static/js/app.js`.

## What the viewer shows

The point cloud is a 2D/3D projection (UMAP or t-SNE) of per-node embeddings for
one run. Two embedding spaces can exist per run and are chosen with the
**Embedding space** selector in the bottom-right panel:

- **Featurization** — the run's raw per-node features, straight from whatever
  featurization method it used (word2vec, doc2vec, fasttext, flash, …). It is
  labelled with the actual method, not hardcoded to "Word2Vec".
- **GNN Encoder** — the trained encoder's node embeddings. The epoch selector
  picks which epoch's model; the default is the **best epoch** (highest ADP in
  the manifest). Embeddings are computed over the **entire test/evaluation
  split, across all its time windows** — train/val nodes are not embedded.

## Inspecting a node & graph tools

Click a point to select it; the **Selected Node** panel (top-right, under the
legend) shows its type, label, anomaly score, path, and a K=20 neighbourhood
summary. From there:

- **See Edges** — every edge incident to the node, with direction, edge type
  (relation), time window, and the neighbour's score/path. Grouping collapses
  the same edge seen across time windows; **Export CSV** dumps the table.
- **Extract Causal Subgraph** — a time-respecting provenance trace from the
  node: it walks out-edges forward in time and in-edges backward, gathering the
  events that could have influenced, or been influenced by, this node (capped at
  10 000). The **Chronological Table** lists them oldest-first with scores; the
  **Directed Graph View** draws the origin plus the malicious nodes reached.
- **Show Attack Graph** (overlay) — draws edges between malicious nodes directly
  in the embedding space, coloured by activation time.
- **Campaign Attack Graph** — a separate force-directed provenance view of the
  attack (entities as nodes, events as labelled directed edges) with a hop
  selector for surrounding context.

These graph tools read the run's adjacency (`*_adj.json`), but the browser never
downloads it — the server answers per-node slices (`/api/neighbors`,
`/api/causal`, `/api/attack_pairs`, `/api/campaign`). Newly generated runs **pre-build these caches at export time** (the binary point
buffer `*.webcache_v4.bin` and the CSR index `*_adj.json.idxcache_v1.npz`), so
the server never parses the 100s-of-MB `points.json`/`adj.json` at all — the
first open is already fast. For older runs generated before this, the server
builds the CSR index on first touch and reuses it forever (re-parsing 350 MB of
JSON takes ~17 s, but the index reloads in ~0.2 s and survives restarts), and it
**warms in a background thread the moment a run is opened** (the adjacency index
and metadata store, plus the sibling epoch/embedding caches), so switching epochs
and clicking "See Edges" are instant. The campaign graph is **not** warmed — it
is a pure-Python build that would hold the GIL and starve the buffer requests the
user is waiting on; it is cached on first click instead.

## How it works

The server parses each `embedding_viz_*_points.json` once and caches these files
next to it (cache version `v4`):

- a binary point buffer (`*.webcache_v4.bin`) for direct GPU upload, holding
  `float16` positions (per hop), temporal attributes, and packed node id + flags
  (no colour column);
- a binary metadata store (`*.store.npz`) — numeric columns plus string columns
  as a byte-blob + offsets — queried per node / full-text server-side, never
  bulk-shipped to the browser;
- an info JSON with run statistics, `byte_offsets`, and the detection-cost sweep.

The browser fetches the binary buffer, uploads it to a Three.js `Points` object,
and renders it with a custom shader. Points appear and disappear with their time
window, glow briefly when they first appear, and fade with age, all driven by
per-node lifespans and a single time uniform. The score distribution is rendered
on the server with matplotlib and served as an image.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/runs` | discover runs under the artifacts root |
| `GET /api/run?file=<points.json>` | run descriptor: n, hops, stats, detection cost, embeddings, epochs |
| `GET /api/buffer?file=` | binary point buffer (cached; Range requests stream it in chunks) |
| `GET /api/node?file=&ids=` | full metadata for a few node ids (on-demand, on selection) |
| `GET /api/search?file=&q=` | node ids matching a query (id / path / cmd), server-side scan |
| `POST /api/filter` `{file, terms}` | node ids matching any of many terms (CSV filter) |
| `GET /api/neighbors?file=&node=` | one node's incident edges ("See Edges"), metadata-enriched |
| `GET /api/causal?file=&node=` | causal subgraph from a node (server-side trace) |
| `GET /api/attack_pairs?file=` | malicious↔malicious edge pairs (3D attack overlay) |
| `GET /api/campaign?file=` | campaign attack graph (nodes + links) |
| `GET /api/scoredist?file=` | anomaly-score distribution as a PNG |
| `POST /api/export`, `GET /api/export/status`, `GET /api/export/stream`, `POST /api/export/cancel` | generation control and live log stream |

All `file` parameters are validated to live under the artifacts root.

## Features

- 3D point cloud with temporal playback, plus a flat 2D mode
- hops scrubbing, epoch selector, and Word2Vec / Encoder switching
- filters (benign, detected, undetected), text search, and CSV node filter
- click to select, with a node inspector and K=20 neighbourhood analysis
- global statistics and a detection-cost panel
- anomaly-score distribution figure
- temporal trajectories, attack-graph overlay on the embedding space,
  discrimination heatmap, and false-positive overlays (full recall and full
  campaign)
- campaign attack graph: a separate force-directed provenance view (entities as
  nodes, events as labeled directed edges) with a hop selector for context
- causal subgraph (chronological table and directed graph)
- anomalous edges with edge type, direction filter, time-window grouping, and CSV export

## Vendored libraries (offline)

`static/vendor/`: `three.min.js` (r128), `OrbitControls.js`, and `d3.v4.min.js`
(used by the campaign attack graph).
