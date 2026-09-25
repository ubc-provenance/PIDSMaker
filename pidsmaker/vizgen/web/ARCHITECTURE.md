# PIDSMaker Web Visualizer: Architecture and Reproducibility Manual

This document describes what the web visualizer is, how it works end to end, and
how to run and view it, so it can be operated and extended without prior context.
See `README.md` for quick commands; this file is the detailed reference.

---

## 1. Overview

A browser-based 3D viewer for the per-node embeddings PIDSMaker produces. It
projects 128-D Word2Vec and GNN-encoder embeddings to 3D with UMAP and renders
the resulting point cloud interactively, scaling from small runs to 20M+ nodes
through a tiered LOD streaming pipeline (section 6), with detection overlays,
temporal playback, node inspection, causal tracing, and the anomaly-score
distribution.

Design goals:

1. Independent of the pipeline. Visualization is never a pipeline stage. You
   generate the data and pick a run from the artifacts folder yourself.
2. Headless server, browser client. Rendering happens in the viewer's browser,
   so the server needs no display libraries.
3. Intended to become one pane inside a larger PIDSMaker web UI.

---

## 2. The three stages

Heavy work is done once at generation time so the server and browser stay light.

| Stage | Runs where | GPU | Role |
|---|---|---|---|
| 1. Generate (`exporter.py`) | server | Yes (GNN + cuML UMAP) | writes `*_points.json` and `*_adj.json` into a run's `viz/` |
| 2. Serve (`viz_server.py`) | server | No (CPU) | parses JSON once into a cached binary buffer; serves bytes and the SPA |
| 3. View (Three.js SPA) | browser | Yes, the client's GPU via WebGL | uploads the buffer to the GPU, renders, handles interaction |

"The viewer is CPU-only" refers to the server: it needs no GPU. The 3D rendering
still uses a GPU, specifically the GPU of whichever machine opens the browser.

---

## 3. Running and viewing

### Generate data for a run (stage 1, requires the GPU machine)

```bash
# In-browser: Run Browser, then Generate or the regenerate button (preferred).
# Or via CLI:
python -m pidsmaker.vizgen.web.export <model> <dataset> --run <eval_dir> --embeddings both
```

`<eval_dir>` is `.../artifacts/evaluation/evaluation/<hash>/<dataset>`.

### Start the server (stage 2)

```bash
python -m pidsmaker.vizgen.web.viz_server --host 0.0.0.0 --port 5000
```

The artifacts root is `/home/artifacts` if present, otherwise
`$PIDS_ARTIFACTS_DIR`, otherwise `<repo>/artifacts`.

### View it (stage 3)

You already have an SSH session into the server to work with PIDSMaker. Viewing
adds one server shell for the viewer and one tunnel on your local machine. Three
steps:

1. On the server, in the PIDSMaker container shell, start the viewer and leave it
   running:
   ```bash
   python -m pidsmaker.vizgen.web.viz_server --host 0.0.0.0 --port 5000
   ```

2. On your local machine, open a new terminal and create an SSH tunnel to the
   server. This command runs locally, not in a server shell. Leave it running
   (`-N` opens no remote shell, only the tunnel):
   ```bash
   ssh -N -L 5000:localhost:5000 <user>@<server>
   ```
   `localhost` here is resolved on the server, so this works when port 5000 is
   reachable on the server host. If PIDSMaker runs in Docker and 5000 is not
   published to the host, target the container IP instead (find it with
   `docker inspect <container>`):
   ```bash
   ssh -N -L 5000:<container-ip>:5000 <user>@<server>
   ```

3. Open `http://localhost:5000` in your browser on the local machine.

Notes:

- `compose-pidsmaker.yml` maps `5000:5000`, so once the container is recreated,
  the host-`localhost` form in step 2 works. Until then, use the container IP.
- The tunnel is secure and exposes nothing. Publishing the port for direct
  `http://<server-ip>:5000` access is optional and for LAN/VPN only; keep it
  firewalled, as there is no authentication.
- The app reopens the last run you viewed (stored in localStorage). If that run
  is gone it loads the newest run that has viz data, otherwise it opens the Run
  Browser.
- Rendering is client-side: the 3D view uses the GPU of the machine that opens
  the browser, since the tunnel only transfers data. See section 10.

---

## 4. Load order

```
browser GET /                      -> index.html
  loads vendor (three, OrbitControls, d3) + data.js, scene.js,
        dialogs.js, campaign.js, jobs.js, ui.js, app.js
app.js DOMContentLoaded:
  Scene.init(canvas)               -> WebGL renderer, camera, animate loop
  UI.init()                        -> build panel handlers, Jobs.reattachOnLoad
  Data.getRuns()  -> /api/runs     -> pick last or newest run
  App.loadRun(file):
    /api/run?file=...              -> descriptor (n, hops, stats, epochs, byte_offsets)
    Tier 1, /api/buffer Range requests -> positions + meta byte; each chunk
             renders on arrival, this is the only thing App.loadRun waits on
    Tier 2, /api/buffer Range requests (background) -> attrs + node ids;
             enables the time slider, playback, heatmap, false-positive and
             attack-graph overlays once loaded (S.auxReady)
    App.refresh()                  -> compose colors and visibility, render
  Tier 3, on demand, never fetched up front:
    /api/node       -> per-node string metadata, fetched on selection
    /api/search, /api/filter -> text search and CSV filter, resolved server-side
    /api/neighbors, /api/causal, /api/campaign -> edges, fetched only when asked
```

The streaming step for tiers 1 and 2 starts at `LOD_CHUNK` (250k nodes) and
doubles up to `LOD_CHUNK_MAX` (3M), so a 20M-node run streams in about 12 round
trips instead of 80 (constants in `static/js/app.js`). A run larger than
`LOD_CAP` (25M nodes) loads only its first `LOD_CAP` nodes, in the exporter's
natural order; the cap exists because the browser/GPU is the real limit, not
the server.

Adjacency loads lazily and only per-node (server-side slices for the attack
graph overlay, causal trace, anomalous edges); the browser never downloads the
full `_adj.json`.

---

## 5. Modules

### Server: `viz_server.py` (Flask, depends only on stdlib, numpy, flask)

- Run discovery (`discover_runs`): scans `.../evaluation/evaluation/*/*` and
  `detection/evaluation/*/*` for runs that have a `viz_manifest.json` and/or
  `viz/*_points.json`, and tags each `ready`, `partial`, or `needs_viz`.
- Cache build (`build_cache`): parses a 100 MB+ `_points.json` once and writes
  sidecar files next to it (cache version `v4`):
  - `*.webcache_v4.bin`: packed binary point buffer (section 6),
  - `*.webcache_v4.meta.json.store.npz`: binary metadata store (numeric columns +
    string blobs) queried per node / full-text server-side, never bulk-shipped,
  - `*.webcache_v4.info.json`: n, hops, max_tw, stats, byte_offsets, detection-cost.
  Rebuilt when the source JSON is newer (or the cache version bumps).
- Detection-cost sweep (`compute_detection_cost`): ground-truth and detected
  counts, false positives at the current threshold, and FP for 100% recall and
  100% campaign coverage.
- Export job manager (single job): spawns
  `python -u -m pidsmaker.vizgen.web.export ...` with `PYTHONUNBUFFERED` set so
  `log()` output streams. A daemon thread reads stdout, classifies a coarse
  phase, and fans the lines out to SSE subscribers.
- Score distribution (`render_score_distribution`): the original matplotlib
  figure rendered headless (Agg backend) and returned as a PNG.
- Campaign graph (`build_campaign_graph`): builds the provenance attack graph
  from the run's own `_adj.json` (edges with relation type and direction) and the
  ground-truth node labels/paths from the metadata cache. Hop 0 is the
  malicious<->malicious edges, capped at 500 sampled edges; hops 1/2/3 add
  context via edge sampling (300/200/100). Edges are oriented forward in
  (hop, id) order rather than dropped when discovered backward, which keeps the
  graph acyclic without fragmenting it. Cached in memory; returned as JSON.
- Endpoints: section 7.

### Frontend (`static/`)

| File | Responsibility |
|---|---|
| `index.html` | DOM skeleton (left panel, canvas, overlays); loads scripts |
| `css/app.css` | dark theme, panels, badges, console, status pill |
| `js/data.js` | API client and binary-buffer decode into typed arrays |
| `js/scene.js` | Three.js scene, temporal point shader, cameras, lines, picking, render loop |
| `js/app.js` | app state, `loadRun`, `refresh` (composes colors, visibility, overlays), filters, playback, picking, inspector, KNN |
| `js/ui.js` | panel wiring, Run Browser (badges and filter), Generate modal, live console |
| `js/jobs.js` | export job transport: start, SSE stream, status pill, reattach |
| `js/dialogs.js` | causal subgraph, anomalous edges (edge type, in/out filter, group toggle), score-distribution image |
| `js/campaign.js` | Campaign Attack Graph: renders the server-built provenance graph (D3 force layout, hop selector, zoom, drag) |

### Generation: `exporter.py`, `embed_exporter.py`, `dimensionality_reduction.py`, `html_builder.py`

- `exporter.run_visualization`: for each embedding mode (Word2Vec or encoder
  epochs), loads the model and graphs, extracts embeddings, runs UMAP to 3D, and
  writes points and adjacency via `html_builder.build_html`.
- Config reconstruction: when `--run` is given, merges the run's
  `run_config.yml` so `build_model` rebuilds the exact trained architecture
  (repository defaults may have diverged since the run).
- Graceful failure: if a checkpoint cannot be rebuilt (for example a run with no
  saved config), it logs a clear message and skips the encoder. Word2Vec still
  exports.
- Opt-in training artifacts (`--save_for_viz`, `pidsmaker/main.py`): when a
  training run is started with this flag, `training_loop.py` persists the
  per-epoch model checkpoints and a cached copy of the test graphs for the
  exporter to reuse. It is off by default because these artifacts are large and
  most runs never open the viewer. Without it, the featurization space still
  exports normally, but the GNN-encoder space needs the checkpoints (skipped
  with a warning if absent) and the test graphs are recomputed on demand
  instead of reused.

---

## 6. Data formats

### `*_points.json` (one object per node-instance)

`node_id`, `coords_hops` (or `x`/`y`/`z`), `tw_idx`, `tw_label`,
`label` (0/1), `detection_status` (0 benign, 1 detected, 2 undetected),
`anomaly_score`, `top_edge`, `path`, `type`.

### `*_adj.json`

`{ "<node_id>": [ {nb, t, dir: "in"|"out", et: "EVENT_READ"}, ... ] }`

`et` is the edge relation name, decoded from the model's `edge_type` one-hot via
`get_rel2id`. It is present only in runs regenerated after edge-type support was
added; older data has no `et`, and the UI shows a dash.

### Binary buffer (`/api/buffer`), packed format `v4` (`CACHE_VERSION` in `viz_server.py`)

Little-endian, for n points and H hops, in this order:

1. positions, `float16[H*n*3]`, hop-major (the client expands to `float32`)
2. attrs, `uint16[n*4]`: (tw_idx, tw_start, tw_end, score-as-`float16`)
3. ids, `uint32[n]`: node_id
4. meta, `uint8[n]`: bit-packed `label<<0 | det<<1 | type<<3`

The client slices these by byte offsets from `/api/run`, decodes the `float16`
positions via a 64K lookup table, unpacks the attrs, and derives colour and size
from the meta byte (`nodeRgba`). Packing (`float16` positions, no colour, 1-byte
flags, packed attrs, no size) cuts the buffer from about 64 B/node to about
19 B/node, since the SSH-tunnel transfer is the real cost at 10 to 20 million
nodes. Fields 1 and 4 (positions, meta) are tier 1 of the load order in section
4, fetched first because they are everything the shader needs to render; fields
2 and 3 (attrs, ids) are tier 2, fetched in the background once the cloud is on
screen. Each field is a separate byte range, so splitting the load into tiers
costs no extra requests. Chunk size grows with each fetch (`LOD_CHUNK` to
`LOD_CHUNK_MAX`), so a 20M-node run streams in about 12 round trips instead
of 80.

### `viz_manifest.json`

Written by the pipeline, one per run. Contains `epochs[]` with `epoch`,
`stats_path`, `scores_path`, `adp`, and `disc_score`. Used to label epochs and
pick the best one.

---

## 7. HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/runs` | discovered runs with status, embeddings, epochs, eval_dir |
| `GET /api/run?file=` | run descriptor: n, hops, byte offsets, stats, detection cost, epochs (with adp/disc), config |
| `GET /api/buffer?file=` | packed binary point buffer (cached; Range-chunked) |
| `GET /api/node?file=&ids=` | full metadata for a few node ids (on-demand, on selection) |
| `GET /api/search?file=&q=` | node ids matching a query (id/path/cmd), server-side scan |
| `POST /api/filter` | `{file, terms}` → node ids matching any term (CSV filter) |
| `GET /api/neighbors?file=&node=` | one node's incident edges ("See Edges") |
| `GET /api/causal?file=&node=` | causal subgraph from a node, server-side time-respecting trace |
| `GET /api/attack_pairs?file=` | malicious-to-malicious edge pairs (attack-graph overlay on the embedding space) |
| `GET /api/scoredist?file=` | matplotlib distribution as PNG (cached per run) |
| `GET /api/campaign?file=` | campaign attack graph JSON, built from the run's adjacency and ground-truth labels (hop edge-sampling 500/300/200/100, forward-oriented edges) |
| `POST /api/export` | start a generation job (409 if one is running) |
| `GET /api/export/status` | current job snapshot |
| `GET /api/export/stream` | SSE: `snapshot`, `log`, `phase`, `status` |
| `POST /api/export/cancel` | terminate the current job |

All `file` and `run_dir` parameters are validated to live under the artifacts
root. Generation endpoints are gated by `PIDS_VIZ_ALLOW_EXPORT` (on by default).

---

## 8. Rendering

- Shader (`scene.js`): a single `THREE.Points` with a custom `ShaderMaterial`.
  Per-vertex attributes are `aColor`, `aSize`, `aTwStart`, `aTwEnd`, `aVisible`;
  uniforms are `uTime`, `uDpr`, `uSizeScale`, `uFlatten`, `uDim`. The temporal
  logic runs in GLSL: when `uTime` is non-negative, a node is hidden outside
  `[tw_start, tw_end)`, glows warm on birth, and fades with age. Point size is
  fixed in pixels (crisp), drawn as a round disc.
- `refresh()` (`app.js`) is the compositor. It recomputes visibility (filters,
  search, CSV), display colors (heatmap, false-positive overlays, 2D dimming),
  the white highlight for the selected node, the temporal trajectory, and the
  attack edges, then writes them to the scene. It is the single place that
  composes a frame.
- Occlusion (`scene.js`): the material sets `depthWrite: true`, so occlusion
  follows true 3D depth rather than draw order. A malicious (red) point in
  front of the cluster stays visible, and one actually behind it is correctly
  hidden, regardless of the order points streamed in. The fragment shader
  discards the transparent disc edge and not-yet-grown points, so only solid
  centres write depth.
- Picking: CPU projection of visible points to screen, selecting the nearest
  within about 14 px, backed by a uniform spatial grid built once per hop/cloud
  change so a click resolves without scanning every point. It uses pointer
  events rather than mouse events, because OrbitControls calls
  `preventDefault` and suppresses the latter.
- Performance: a continuous `requestAnimationFrame` loop renders every frame.
  Motion-LOD, adaptive-pixel-ratio, and on-demand (render-only-on-change)
  variants were tried and reverted because they caused point flicker and
  zoom stutter. Drawing a large cloud of semi-transparent points, up to the
  `LOD_CAP` of 25M nodes, is bound by GPU fill rate; see section 10.

---

## 9. Design rationale

- Pre-compute plus binary buffer: parsing a 100 MB+ JSON in the browser is slow.
  The binary blob uploads directly to the GPU, which keeps a multi-million-point
  cloud responsive.
- CPU-only, headless server: it deploys anywhere with no GPU or display needed
  to serve, and rendering offloads to the client.
- Word2Vec is unscored: it is the raw, pre-training featurization with no
  detector. Only the trained per-epoch encoders carry anomaly scores.
- Score distribution via server-side matplotlib: rendering it on the server
  keeps the figure consistent and publication-quality, independent of the
  browser.

---

## 10. Caveats and operational notes

- Edge types appear only in runs regenerated after edge-type support was added.
  Older `_adj.json` files show a dash. Regenerating Word2Vec is sufficient, since
  adjacency edges come from the graph rather than the model.
- Runs without a `run_config.yml` cannot have their encoder rebuilt (the
  architecture is unknown). Their encoder export fails gracefully and Word2Vec
  still works.
- Job state is in memory. Restarting the server drops the current job and kills
  its export subprocess. Only one generation job runs at a time, as it is
  GPU-bound.
- After generation, a run becomes selectable as soon as its `_points.json`
  exists. Reopen the Run Browser; discovery is live.

### Performance and hardware

- Rendering runs on the client GPU, the machine with the browser, not the
  server. The SSH tunnel only transfers data. A workstation GPU renders large
  clouds smoothly; a laptop may stutter on zoom and orbit at millions of
  points. There is no server-side fix for this; for the smoothest experience,
  view on a machine with a strong GPU.
- On laptops, memory is usually the bottleneck rather than the GPU. The point
  buffer is packed at about 19 B/node (section 6), so it stays modest even at
  scale, for example about 190 MB for a 10M-node run. The largest single
  contributor to memory pressure is the adjacency data: it loads lazily, only
  when you open the attack graph, causal subgraph, or anomalous edges, and
  stays in memory for the session once loaded. The server parses `_adj.json`
  once into a CSR cache (a re-parse of a 350 MB adjacency file takes about
  17 s; the cached index then reloads in about 0.2 s and survives restarts), so
  repeat opens of the same run are fast even though the raw JSON is large. If
  you never open those features, only the point buffer is held client-side.
  To recover memory, reload the tab; also close other tabs and applications.
- The point buffer uploads straight to the GPU. The source `_points.json`
  (100 MB+ for large runs) is parsed once on the server, so the client never
  parses it.
