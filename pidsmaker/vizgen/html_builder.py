"""Interactive 2D HTML viewer with time slider for temporal embedding visualization.

Generates a self-contained HTML file with:
- Plotly.js scatter2d (X/Y = UMAP/t-SNE)
- Time window slider to scrub through temporal snapshots
- Click-to-highlight with node dimming
- Node inspector side panel
- Category toggles, search
- Fully offline (Plotly.js bundled inline)
"""

import json
import os
import urllib.request

from pidsmaker.utils.utils import log


def _get_plotly_js():
    """Return Plotly.js source, downloading and caching if needed."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".plotly_cache")
    cache_path = os.path.join(cache_dir, "plotly.min.js")
    if not os.path.exists(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        log("[html_builder] Downloading plotly.min.js for offline bundling...")
        urllib.request.urlretrieve(
            "https://cdn.plot.ly/plotly-2.27.0.min.js", cache_path
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        return f.read()


def build_html(points, edges, node_metadata, title="Embedding Visualization", default_hops=0, out_path=None, is_studio=False):
    plotly_js = _get_plotly_js()

    log(f"[html_builder] Filtering valid IDs for {len(points)} points...")
    valid_ids = {p["node_id"] for p in points}
    adj = {nid: [] for nid in valid_ids}

    log("[html_builder] Building adjacency matrix...")
    for edge in edges:
        u, v, t = int(edge[0]), int(edge[1]), int(edge[2])
        # edge relation is a string label (e.g. "EVENT_RECVMSG"). When edges come
        # in as numpy rows the element is a numpy scalar (numpy.str_/int64) — call
        # .item() to get a native Python str/int that json.dump can serialise.
        et = edge[3] if len(edge) > 3 else ""  # edge relation code
        if hasattr(et, "item"):
            et = et.item()
        if u in valid_ids and v in valid_ids:
            adj[u].append({"nb": v, "t": t, "dir": "out", "et": et})
            adj[v].append({"nb": u, "t": t, "dir": "in", "et": et})

    log("[html_builder] Attaching node metadata...")
    for p in points:
        meta = node_metadata.get(p["node_id"], {})
        path_val = str(meta.get("path", "Unknown")).replace('"', "'")
        cmd_val = str(meta.get("cmd", "")).replace('"', "'")
        type_val = str(meta.get("type", "Unknown")).replace('"', "'")

        if path_val == "None":
            if cmd_val and cmd_val != "None":
                path_val = cmd_val
            elif type_val == "file":
                path_val = "<Anonymous File/Pipe>"

        p["path"] = path_val
        p["type"] = type_val
        p["cmd"] = cmd_val

    if not out_path:
        raise ValueError("out_path must be provided to build_html to save decoupled data files.")

    points_file = out_path.replace('.html', '_points.json')
    adj_file = out_path.replace('.html', '_adj.json')

    max_tw = max((p.get("tw_idx", 0) for p in points), default=0)

    # Compact serialization: round floats, strip empty fields
    log(f"[html_builder] Serializing {len(points)} points to {os.path.basename(points_file)}...")
    compact_points = []
    for p in points:
        cp = {
            "node_id": p["node_id"],
            "coords_hops": [[round(c, 3) for c in hop] for hop in p.get("coords_hops", [[0,0,0]])],
            "tw_idx": p["tw_idx"],
            "tw_label": p["tw_label"],
            "label": p["label"],
            "detection_status": p["detection_status"],
            "anomaly_score": p.get("anomaly_score", 0.0),
            "top_edge": p.get("top_edge", ""),
            "path": p["path"],
            "type": p["type"],
        }
        if p.get("cmd") and p["cmd"] != p["path"]:
            cp["cmd"] = p["cmd"]
        compact_points.append(cp)
    with open(points_file, "w", encoding="utf-8") as f:
        json.dump(compact_points, f, separators=(',', ':'))
    log(f"[html_builder] Points file: {os.path.getsize(points_file) / (1024*1024):.1f} MB")

    # Pre-build the web viewer's binary point cache now, from the in-memory
    # records, so the server never parses this (100s of MB) points.json on first
    # open — the single biggest cold-open cost at scale.
    try:
        from pidsmaker.vizgen.web import viz_server as _vs
        _vs.build_point_cache(compact_points, points_file)
        log("[html_builder] Pre-built binary point cache (instant first open).")
    except Exception as e:
        log(f"[html_builder] point cache pre-build skipped ({type(e).__name__}: {e})")

    # Export a CSV file for offline analysis (Pandas/Excel)
    csv_file = out_path.replace('.html', '_nodes.csv')
    log(f"[html_builder] Exporting nodes to CSV for analysis: {os.path.basename(csv_file)}...")
    import csv as _csv
    with open(csv_file, "w", encoding="utf-8", newline='') as f:
        writer = _csv.writer(f)
        writer.writerow([
            "node_id", "node_type", "path", "cmd",
            "ground_truth_malicious", "detection_status",
            "anomaly_score", "top_edge", "umap_x", "umap_y", "umap_z"
        ])
        for p in compact_points:
            coords = p.get("coords_hops", [[0.0, 0.0, 0.0]])[0]
            x = coords[0] if len(coords) > 0 else 0.0
            y = coords[1] if len(coords) > 1 else 0.0
            z = coords[2] if len(coords) > 2 else 0.0

            writer.writerow([
                p["node_id"],
                p.get("type", ""),
                p.get("path", ""),
                p.get("cmd", ""),
                p.get("label", 0),
                p.get("detection_status", 0),
                p.get("anomaly_score", 0.0),
                p.get("top_edge", ""),
                x, y, z
            ])
    log(f"[html_builder] CSV file: {os.path.getsize(csv_file) / (1024*1024):.1f} MB")

    # Free memory
    del compact_points

    log(f"[html_builder] Serializing adjacency list to {os.path.basename(adj_file)}...")
    with open(adj_file, "w", encoding="utf-8") as f:
        json.dump(adj, f, separators=(',', ':'))
    log(f"[html_builder] Adj file: {os.path.getsize(adj_file) / (1024*1024):.1f} MB")

    # Pre-build the CSR adjacency index too, so See Edges / causal / attack /
    # campaign are instant on first open without parsing this adj.json.
    try:
        from pidsmaker.vizgen.web import viz_server as _vs
        _, _npz, _vocab = _vs._adj_index_paths(points_file)
        _vs.build_adj_index_from_dict(adj, adj_file, _npz, _vocab)
        log("[html_builder] Pre-built CSR adjacency index (instant graph tools).")
    except Exception as e:
        log(f"[html_builder] adj index pre-build skipped ({type(e).__name__}: {e})")

    # Free memory
    del adj

    log(f"[html_builder] Formatting HTML template...")
    html = _HTML_TEMPLATE.format(
        title=title,
        plotly_js=plotly_js,
        points_url=os.path.basename(points_file),
        adj_url=os.path.basename(adj_file),
        max_tw=max_tw,
        num_points=len(points),
        default_hops=default_hops,
        studio_css="block" if is_studio else "none",
        is_studio=str(is_studio).lower(),
    )

    log("[html_builder] HTML template formatting complete.")
    return html


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',system-ui,sans-serif;background:#000000;color:#e0e0e8;display:flex;height:100vh;overflow:hidden}}
#plot-container{{flex:3;position:relative;height:100%;display:flex;flex-direction:column}}
#plotly-div{{flex:1;width:100%}}
#slider-bar{{padding:10px 20px;background:rgba(15,15,25,0.95);border-top:1px solid rgba(100,180,255,0.1);display:flex;align-items:center;gap:14px}}
#slider-bar label{{font-size:12px;color:rgba(255,255,255,0.6);white-space:nowrap}}
#tw-slider{{flex:1;accent-color:#64b4ff}}
#tw-display{{font-size:13px;font-weight:600;color:#64b4ff;min-width:80px;text-align:center}}
#side-panel{{flex:0 0 340px;background:rgba(15,15,25,0.95);border-left:1px solid rgba(100,180,255,0.12);padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:16px;backdrop-filter:blur(12px)}}
h2{{font-size:18px;font-weight:600;color:#fff;letter-spacing:-0.3px}}
h3{{font-size:13px;font-weight:500;color:rgba(255,255,255,0.6);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}}
.card{{background:rgba(25,25,40,0.8);border:1px solid rgba(100,180,255,0.08);border-radius:10px;padding:14px}}
.control-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;font-size:13px}}
.control-row:last-child{{margin-bottom:0}}
.control-row label{{color:rgba(255,255,255,0.7)}}
input[type="range"]{{width:100px;accent-color:#64b4ff}}
input[type="checkbox"]{{accent-color:#64b4ff;width:16px;height:16px}}
input[type="text"]{{background:rgba(255,255,255,0.06);border:1px solid rgba(100,180,255,0.15);border-radius:6px;padding:6px 10px;color:#e0e0e8;font-size:13px;width:100%;outline:none;font-family:inherit}}
input[type="text"]:focus{{border-color:rgba(100,180,255,0.4)}}
.node-info{{font-size:12px;line-height:1.7;word-break:break-all}}
.node-info b{{color:rgba(100,180,255,0.9);font-weight:500}}
.neighbor-item{{padding:6px 8px;margin-bottom:4px;background:rgba(255,255,255,0.03);border-radius:6px;font-size:11px;cursor:pointer;transition:background 0.15s}}
.neighbor-item:hover{{background:rgba(100,180,255,0.1)}}
.st-benign{{border-left:3px solid rgba(100,200,150,0.6)}}
.st-detected{{border-left:3px solid rgba(255,80,80,0.8)}}
.st-undetected{{border-left:3px solid rgba(255,165,0,0.8)}}
#header-bar{{position:absolute;top:12px;left:16px;z-index:10;font-size:12px;color:rgba(255,255,255,0.5);background:rgba(10,10,15,0.7);padding:6px 14px;border-radius:8px;backdrop-filter:blur(8px);pointer-events:none}}
.empty-state{{color:rgba(255,255,255,0.35);font-size:12px;font-style:italic}}
#inspector-scroll{{overflow-y:auto;flex-grow:1}}
.trajectory-info{{font-size:11px;color:rgba(100,180,255,0.7);margin-top:4px}}
.tw-btn{{background:rgba(100,180,255,0.15);border:1px solid rgba(100,180,255,0.25);color:#64b4ff;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;font-family:inherit}}
.tw-btn:hover{{background:rgba(100,180,255,0.25)}}
.tw-btn.active{{background:rgba(100,180,255,0.35);border-color:#64b4ff}}
#play-btn{{background:rgba(100,180,255,0.2);border:1px solid rgba(100,180,255,0.3);color:#64b4ff;border-radius:6px;padding:4px 12px;font-size:13px;cursor:pointer;font-family:inherit}}
#play-btn:hover{{background:rgba(100,180,255,0.3)}}
#floating-controls{{position:absolute;top:12px;right:16px;z-index:10;display:flex;gap:8px}}
.icon-btn{{background:rgba(15,15,25,0.85);border:1px solid rgba(100,180,255,0.2);color:#e0e0e8;border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;backdrop-filter:blur(8px);transition:all 0.2s}}
.icon-btn:hover{{background:rgba(100,180,255,0.2);border-color:#64b4ff}}
.toggle-switch{{display:flex;align-items:center;gap:8px;font-size:13px;color:rgba(255,255,255,0.7)}}
#loading-overlay{{position:absolute;top:0;left:0;right:0;bottom:0;background:#000000;z-index:999;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#64b4ff;font-size:18px;font-weight:600;gap:12px}}
.spinner{{width:30px;height:30px;border:3px solid rgba(100,180,255,0.2);border-top-color:#64b4ff;border-radius:50%;animation:spin 1s linear infinite}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div id="plot-container">
  <div id="loading-overlay"><div class="spinner"></div><div id="loading-text">Loading WebGL Visualization Engine...</div></div>
  <div id="header-bar">{title} &mdash; <span id="point-count">{num_points}</span> points &bull; <span id="tw-count">{max_tw}+1</span> time windows</div>
  <div id="floating-controls">
    <button class="icon-btn" onclick="resetView()" title="Reset Camera View">&#8635; Reset View</button>
  </div>
  <div id="plotly-div"></div>
  <div id="slider-bar">
    <button id="play-btn" onclick="togglePlay()">&#9654;</button>
    <label>Time Window:</label>
    <input type="range" id="tw-slider" min="-1" max="{max_tw}" value="-1" step="1">
    <span id="tw-display">All</span>
  </div>
</div>
<div id="side-panel">
  <h2>Node Inspector</h2>
  <div class="card">
    <h3>Controls</h3>
    <div class="control-row">
      <label for="hop-slider">Hops (<span id="hop-val">{default_hops}</span>):</label>
      <input type="range" id="hop-slider" min="0" max="3" value="{default_hops}">
    </div>
    <div class="control-row" style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
      <label class="toggle-switch">
        <input type="checkbox" id="mode-toggle" onchange="toggleMode()" checked>
        <span>3D Temporal Mode</span>
      </label>
    </div>
  </div>
  <div class="card" id="umap-settings" style="display:{studio_css}">
    <h3>UMAP Settings (Studio)</h3>
    <div class="control-row">
      <label>Neighbors:</label>
      <input type="range" id="studio-knn" min="5" max="50" value="15" oninput="document.getElementById('knn-val').innerText=this.value">
      <span id="knn-val" style="color:#64b4ff">15</span>
    </div>
    <div class="control-row">
      <label>Min Dist:</label>
      <input type="range" id="studio-md" min="0" max="1" step="0.05" value="0.1" oninput="document.getElementById('md-val').innerText=this.value">
      <span id="md-val" style="color:#64b4ff">0.1</span>
    </div>
    <button id="recalc-btn" class="icon-btn" style="width:100%;margin-top:10px;background:#1d4ed8;border:none" onclick="recalcUmap()">Recalculate UMAP</button>
  </div>
  <div class="card">
    <h3>Filter</h3>
    <div class="control-row"><label><input type="checkbox" id="cb-benign" checked> Benign</label></div>
    <div class="control-row"><label><input type="checkbox" id="cb-detected" checked> Detected</label></div>
    <div class="control-row"><label><input type="checkbox" id="cb-undetected" checked> Undetected</label></div>
  </div>
  <div class="card">
    <h3>Search</h3>
    <input type="text" id="search-box" placeholder="Node ID or path...">
  </div>
  <div id="inspector-scroll">
    <div id="inspector-content"></div>
  </div>
</div>
<script>{plotly_js}</script>
<script>
let points = [];
let adj = {{}};
const maxTW = {max_tw};

// Index: node_id -> list of point indices
const nodeIndex = {{}};
let allTWs = [];

async function _loadData() {{
  const txt = document.getElementById('loading-text');
  try {{
    txt.innerText = "Downloading point data...";
    const pRes = await fetch('{points_url}');
    const pTotal = parseInt(pRes.headers.get('content-length') || '0');
    if (pTotal > 0) {{
      const reader = pRes.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {{
        const {{done, value}} = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        txt.innerText = `Downloading points: ${{(received/1024/1024).toFixed(1)}} / ${{(pTotal/1024/1024).toFixed(1)}} MB`;
      }}
      const blob = new Blob(chunks);
      const text = await blob.text();
      txt.innerText = "Parsing point data...";
      points = JSON.parse(text);
    }} else {{
      points = await pRes.json();
    }}
    // Ensure cmd field exists on all points and unpack coords_hops into p.x/p.y/p.z
    points.forEach(p => {{
      if (!p.cmd) p.cmd = '';
      const c = (p.coords_hops && p.coords_hops[0]) || [0,0,0];
      p.x = c[0]; p.y = c[1]; p.z = c[2];
    }});

    txt.innerText = "Downloading graph topology...";
    const aRes = await fetch('{adj_url}');
    adj = await aRes.json();

    txt.innerText = `Indexing ${{points.length.toLocaleString()}} nodes...`;

    points.forEach((p,i) => {{
      if (!nodeIndex[p.node_id]) nodeIndex[p.node_id] = [];
      nodeIndex[p.node_id].push(i);
    }});

    allTWs = [...new Set(points.map(p => p.tw_idx))].sort((a,b) => a-b);
    document.getElementById('tw-count').textContent = allTWs.length;

    txt.innerText = "Rendering...";
    // Use requestAnimationFrame to let the browser paint the status before heavy render
    requestAnimationFrame(() => {{
      _initApp();
      document.getElementById('loading-overlay').style.display = 'none';
    }});
  }} catch (e) {{
    txt.innerText = "Failed to load data: " + e.message;
    console.error(e);
  }}
}}

async function recalcUmap() {{
  const knn = document.getElementById('studio-knn').value;
  const md = document.getElementById('studio-md').value;
  const btn = document.getElementById('recalc-btn');
  const overlay = document.getElementById('loading-overlay');
  const txt = document.getElementById('loading-text');

  btn.disabled = true;
  overlay.style.display = 'flex';
  txt.innerText = "Sending recalculation request...";

  try {{
    const res = await fetch('/api/recalculate', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ n_neighbors: parseInt(knn), min_dist: parseFloat(md) }})
    }});

    if (res.ok) {{
      pollStudioStatus();
    }} else {{
      txt.innerText = "Error starting recalculation.";
      setTimeout(() => overlay.style.display = 'none', 2000);
      btn.disabled = false;
    }}
  }} catch(e) {{
    txt.innerText = "Connection error.";
    setTimeout(() => overlay.style.display = 'none', 2000);
    btn.disabled = false;
  }}
}}

function pollStudioStatus() {{
  const txt = document.getElementById('loading-text');
  const overlay = document.getElementById('loading-overlay');
  const btn = document.getElementById('recalc-btn');

  fetch('/api/status').then(r => r.json()).then(data => {{
    if (data.status === 'running') {{
      txt.innerText = data.progress || "Calculating UMAP...";
      setTimeout(pollStudioStatus, 500);
    }} else if (data.status === 'idle') {{
      txt.innerText = "Reloading new data...";
      // Reload points dynamically
      fetch('{points_url}').then(r => r.json()).then(newPoints => {{
        points = newPoints;
        points.forEach(p => {{
          if (!p.cmd) p.cmd = '';
          const c = (p.coords_hops && p.coords_hops[0]) || [0,0,0];
          p.x = c[0]; p.y = c[1]; p.z = c[2];
        }});
        nodeIndex = {{}};
        points.forEach((p,i) => {{
          if (!nodeIndex[p.node_id]) nodeIndex[p.node_id] = [];
          nodeIndex[p.node_id].push(i);
        }});
        requestAnimationFrame(() => {{
          _initApp();
          overlay.style.display = 'none';
          btn.disabled = false;
        }});
      }});
    }} else {{
      txt.innerText = "Error: " + data.progress;
      setTimeout(() => overlay.style.display = 'none', 3000);
      btn.disabled = false;
    }}
  }});
}}

// We wrap the remaining synchronous initialization code into _initApp
function _initApp() {{


const C = {{
  benign:         'rgba(50, 200, 100, 0.25)',   // Green, transparent
  det_process:    'rgba(255, 51, 51, 0.7)',     // Red
  det_netflow:    'rgba(51, 102, 255, 0.7)',    // Blue
  det_file:       'rgba(255, 204, 0, 0.7)',     // Yellow
  undet_process:  'rgba(255, 153, 51, 0.7)',    // Orange
  undet_netflow:  'rgba(51, 204, 255, 0.7)',    // Cyan
  undet_file:     'rgba(255, 255, 102, 0.7)',   // Light Yellow
  benignHi:       'rgba(51, 204, 102, 0.9)',
  dimmed:         'rgba(40, 45, 50, 0.05)',
  selected:       'rgba(255, 255, 255, 1.0)',
}};

function getColor(p) {{
  if (p.label === 1) {{
    const t = (p.type || '').toLowerCase();
    if (p.detection_status === 1 || p.detection_status === 0) {{ // Detected or W2V GT
      if (t.includes('process') || t.includes('subject')) return C.det_process;
      if (t.includes('netflow')) return C.det_netflow;
      if (t.includes('file')) return C.det_file;
      return C.det_process; // fallback
    }} else {{ // Undetected
      if (t.includes('process') || t.includes('subject')) return C.undet_process;
      if (t.includes('netflow')) return C.undet_netflow;
      if (t.includes('file')) return C.undet_file;
      return C.undet_process; // fallback
    }}
  }}
  return C.benign;
}}
function getSize(p) {{
  return 5;
}}

let currentTW = -1; // -1 = all
let selectedNodeId = null;
let activeTraceNodes = null;
let playing = false;
let playTimer = null;
let is3D = true;

function getVisiblePoints() {{
  return points.filter(p => {{
    if (currentTW >= 0 && p.tw_idx !== currentTW) return false;
    return true;
  }});
}}

function buildTraces(dimNodeId) {{
  const vis = getVisiblePoints();
  const cats = [
    {{name:'Benign', color: C.benign, filter: p => p.label===0, show: document.getElementById('cb-benign').checked}},
    {{name:'Det. Process', color: C.det_process, filter: p => p.label===1 && (p.detection_status===1||p.detection_status===0) && ((p.type || '').toLowerCase().includes('process') || (p.type || '').toLowerCase().includes('subject')), show: document.getElementById('cb-detected').checked}},
    {{name:'Det. Netflow', color: C.det_netflow, filter: p => p.label===1 && (p.detection_status===1||p.detection_status===0) && (p.type || '').toLowerCase().includes('netflow'), show: document.getElementById('cb-detected').checked}},
    {{name:'Det. File', color: C.det_file, filter: p => p.label===1 && (p.detection_status===1||p.detection_status===0) && (p.type || '').toLowerCase().includes('file'), show: document.getElementById('cb-detected').checked}},
    {{name:'Undet. Process', color: C.undet_process, filter: p => p.label===1 && p.detection_status===2 && ((p.type || '').toLowerCase().includes('process') || (p.type || '').toLowerCase().includes('subject')), show: document.getElementById('cb-undetected').checked}},
    {{name:'Undet. Netflow', color: C.undet_netflow, filter: p => p.label===1 && p.detection_status===2 && (p.type || '').toLowerCase().includes('netflow'), show: document.getElementById('cb-undetected').checked}},
    {{name:'Undet. File', color: C.undet_file, filter: p => p.label===1 && p.detection_status===2 && (p.type || '').toLowerCase().includes('file'), show: document.getElementById('cb-undetected').checked}},
  ];
  const traces = [];
  const isDimmed = dimNodeId !== null;
  const traceType = is3D ? 'scatter3d' : 'scattergl';

  const hops = parseInt(document.getElementById('hop-slider').value) || 0;
  const activeNeighbors = isDimmed && hops > 0 ? getKHopNeighbors(dimNodeId, hops) : [];

  cats.forEach(cat => {{
    const pts = vis.filter(cat.filter);
    if (pts.length === 0) return;

    const colors = pts.map(p => {{
      if (activeTraceNodes) {{
        return activeTraceNodes.has(p.node_id) ? cat.color : C.dimmed;
      }}
      return cat.color;
    }});

    const opacities = pts.map(p => {{
      if (activeTraceNodes) {{
        return activeTraceNodes.has(p.node_id) ? 1.0 : 0.05;
      }}
      return cat.name === 'Benign' ? (is3D ? 0.25 : 0.3) : (is3D ? 0.7 : 0.75);
    }});

    const trace = {{
      x: pts.map(p=>p.x), y: pts.map(p=>p.y),
      customdata: pts.map(p => p.node_id),
      mode:'markers', type: traceType, name:cat.name,
      visible: cat.show ? true : 'legendonly',
      hoverinfo:'text',
      marker: {{
        size: cat.name === 'Benign' ? (is3D ? 2 : 2.5) : (is3D ? 3.5 : 4),
        color: colors,
        line: {{width:0}},
        opacity: opacities,
      }},
    }};
    // Only generate hover text for small traces or malicious — saves huge memory for benign
    if (pts.length < 50000 || cat.name !== 'Benign') {{
      trace.text = pts.map(p => `ID:${{p.node_id}} | ${{p.type}} | ${{(p.path||'').substring(0,40)}}<br>TW:${{p.tw_label}}`);
    }} else {{
      trace.hoverinfo = 'skip'; // Skip hover for huge benign traces
    }}
    if (is3D) trace.z = pts.map(p=>p.z);
    traces.push(trace);
  }});

  // If a node is selected, add ghost markers for ALL its TW appearances (even filtered ones)
  if (dimNodeId !== null) {{
    const allAppearances = (nodeIndex[dimNodeId] || []).map(i => points[i]);
    if (allAppearances.length > 1) {{
      const sorted = allAppearances.sort((a,b) => a.tw_idx - b.tw_idx);
      // Ghost trail dots
      const lineTrace = {{
        x: sorted.map(p=>p.x), y: sorted.map(p=>p.y),
        text: sorted.map(p => `TW:${{p.tw_label}} pos(${{p.x.toFixed(3)}}, ${{p.y.toFixed(3)}})`),
        mode:'markers+lines', type: traceType, name:'Trajectory',
        showlegend:false, hoverinfo:'text',
        line: {{color:'rgba(100,180,255,0.4)', width:2, dash:'dot'}},
        marker: {{size:is3D?5:8, color:'rgba(100,180,255,0.8)', symbol:'diamond', line:{{width:0}}}},
      }};
      if (is3D) lineTrace.z = sorted.map(p=>p.z);
      traces.push(lineTrace);
    }}
  }}


  // Subtle XYZ axis lines at the origin for 3D spatial reference
  if (is3D) {{
    const axLen = 15;
    traces.push({{ x: [-axLen, axLen], y: [0, 0], z: [0, 0], mode: 'lines', line: {{color: 'rgba(255,80,80,0.25)', width: 1.5}}, showlegend: false, hoverinfo: 'skip', type: 'scatter3d' }});
    traces.push({{ x: [0, 0], y: [-axLen, axLen], z: [0, 0], mode: 'lines', line: {{color: 'rgba(80,255,80,0.25)', width: 1.5}}, showlegend: false, hoverinfo: 'skip', type: 'scatter3d' }});
    traces.push({{ x: [0, 0], y: [0, 0], z: [-axLen, axLen], mode: 'lines', line: {{color: 'rgba(80,80,255,0.25)', width: 1.5}}, showlegend: false, hoverinfo: 'skip', type: 'scatter3d' }});
  }}

  // Mother Node Star
  if (dimNodeId !== null && vis.length > 0) {{
    const motherPt = vis.find(p => p.node_id === dimNodeId);
    if (motherPt) {{
      const starTrace = {{
        x: [motherPt.x], y: [motherPt.y],
        text: [`<b>[INITIAL STATE]</b><br>ID:${{motherPt.node_id}} | ${{motherPt.type}} | ${{motherPt.path.substring(0,40)}}`],
        mode: 'markers', type: traceType, name: 'Selected Node',
        showlegend: false, hoverinfo: 'text',
        marker: {{ size: is3D ? 10 : 15, color: '#00ffff', symbol: is3D ? 'diamond' : 'star', line: {{color: 'white', width: 1}} }}
      }};
      if (is3D) starTrace.z = [motherPt.z];
      traces.push(starTrace);
    }}
  }}

  return traces;
}}

function getLayout() {{
  if (is3D) {{
    return {{
      scene: {{
        xaxis: {{visible: false, showgrid: false, zeroline: false}},
        yaxis: {{visible: false, showgrid: false, zeroline: false}},
        zaxis: {{visible: false, showgrid: false, zeroline: false}},
        bgcolor: '#000000',
        camera: typeof cameraState !== 'undefined' ? cameraState : {{eye: {{x: 1.25, y: 1.25, z: 1.25}}}}
      }},
      plot_bgcolor:'#000000',
      paper_bgcolor:'#000000',
      font: {{color:'#e0e0e8', family:'Inter'}},
      showlegend:true,
      legend: {{x:0.01, y:0.99, bgcolor:'rgba(15,15,25,0.7)', bordercolor:'rgba(100,180,255,0.1)', borderwidth:1, font:{{size:11}}}},
      margin: {{l:0,r:0,t:0,b:0}},
      dragmode: 'turntable'
    }};
  }} else {{
    return {{
      xaxis: {{visible: false, showgrid: false, zeroline: false}},
      yaxis: {{visible: false, showgrid: false, zeroline: false, scaleanchor:'x'}},
      plot_bgcolor:'#000000',
      paper_bgcolor:'#000000',
      font: {{color:'#e0e0e8', family:'Inter'}},
      showlegend:true,
      legend: {{x:0.01, y:0.99, bgcolor:'rgba(15,15,25,0.7)', bordercolor:'rgba(100,180,255,0.1)', borderwidth:1, font:{{size:11}}}},
      margin: {{l:0,r:0,t:0,b:0}},
      dragmode:'pan',
    }};
  }}
}}

const config = {{scrollZoom:true, responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage','lasso2d','select2d','resetCameraDefault3d','hoverClosest3d']}};
let cameraState = {{eye: {{x: 1.5, y: 1.5, z: 0.5}}, center: {{x:0, y:0, z:0}}, up: {{x:0, y:0, z:1}}}};
let autoRotateTimer = null;
let isInteracting = false;

// Precompute Global Stats
const uniqueNodes = new Map();
points.forEach(p => {{
  if (!uniqueNodes.has(p.node_id)) {{
    uniqueNodes.set(p.node_id, p);
  }}
}});
const uNodes = Array.from(uniqueNodes.values());
const globalStats = {{
  total: uNodes.length,
  benign: uNodes.filter(p => p.label === 0).length,
  malicious: uNodes.filter(p => p.label === 1).length,
  mal_process: uNodes.filter(p => p.label === 1 && ((p.type||'').toLowerCase().includes('process') || (p.type||'').toLowerCase().includes('subject'))).length,
  mal_netflow: uNodes.filter(p => p.label === 1 && (p.type||'').toLowerCase().includes('netflow')).length,
  mal_file: uNodes.filter(p => p.label === 1 && (p.type||'').toLowerCase().includes('file')).length,
}};

function getGlobalStatsHtml() {{
  let html = '<div class="card">';
  html += `<h3>Global Statistics</h3>`;
  html += `<p style="margin-top:0;font-size:13px;color:#aaa;">Overall unique nodes present in this dataset projection.</p>`;
  html += `<div style="display:flex; justify-content:space-between; margin-bottom:4px"><span>Total Nodes:</span> <b>${{globalStats.total}}</b></div>`;
  html += `<div style="display:flex; justify-content:space-between; margin-bottom:4px; color:${{C.benign}}"><span>Benign Nodes:</span> <b>${{globalStats.benign}}</b></div>`;
  html += `<div style="display:flex; justify-content:space-between; margin-bottom:12px; color:${{C.det_process}}"><span>Malicious Nodes:</span> <b>${{globalStats.malicious}}</b></div>`;
  if (globalStats.malicious > 0) {{
    html += `<div style="font-size:12px; margin-left:12px; border-left:2px solid #555; padding-left:8px;">`;
    html += `<div style="display:flex; justify-content:space-between; margin-bottom:4px; color:${{C.det_process}}"><span>Processes:</span> <span>${{globalStats.mal_process}}</span></div>`;
    html += `<div style="display:flex; justify-content:space-between; margin-bottom:4px; color:${{C.det_netflow}}"><span>Netflows:</span> <span>${{globalStats.mal_netflow}}</span></div>`;
    html += `<div style="display:flex; justify-content:space-between; margin-bottom:4px; color:${{C.det_file}}"><span>Files:</span> <span>${{globalStats.mal_file}}</span></div>`;
    html += `</div>`;
  }}
  html += '</div>';
  return html;
}}

Plotly.newPlot('plotly-div', buildTraces(null), getLayout(), config);
const plotDiv = document.getElementById('plotly-div');
document.getElementById('inspector-content').innerHTML = getGlobalStatsHtml();

function startAutoRotate() {{
  if (autoRotateTimer) clearInterval(autoRotateTimer);
  autoRotateTimer = setInterval(() => {{
    if (!is3D || isInteracting) return;
    const angle = 0.003;
    const cx = cameraState.eye.x;
    const cy = cameraState.eye.y;
    cameraState = {{
      ...cameraState,
      eye: {{
        x: cx * Math.cos(angle) - cy * Math.sin(angle),
        y: cx * Math.sin(angle) + cy * Math.cos(angle),
        z: cameraState.eye.z
      }}
    }};
    Plotly.relayout('plotly-div', {{'scene.camera': cameraState}});
  }}, 50);
}}

function stopAutoRotate() {{
  if (autoRotateTimer) {{
    clearInterval(autoRotateTimer);
    autoRotateTimer = null;
  }}
}}

plotDiv.on('plotly_relayout', function(eventData) {{
  if (eventData['scene.camera']) {{
    cameraState = eventData['scene.camera'];
  }}
}});

plotDiv.addEventListener('mousedown', () => {{ isInteracting = true; stopAutoRotate(); }});
plotDiv.addEventListener('mouseup', () => {{ isInteracting = false; }});
plotDiv.addEventListener('wheel', () => {{ isInteracting = true; stopAutoRotate(); }});

plotDiv.on('plotly_click', function(data) {{
  if (!data.points.length) return;
  const nid = data.points[0].customdata;
  if (nid === undefined || nid === null) return;
  selectNode(nid);
}});

function selectNode(nid) {{
  selectedNodeId = nid;
  Plotly.react('plotly-div', buildTraces(nid), getLayout(), config);
  updateInspector(nid);
}}

function clearSelection() {{
  selectedNodeId = null;
  activeTraceNodes = null;
  Plotly.react('plotly-div', buildTraces(null), getLayout(), config);
  document.getElementById('inspector-content').innerHTML = getGlobalStatsHtml();
}}

function refresh() {{
  Plotly.react('plotly-div', buildTraces(selectedNodeId), getLayout(), config);
  if (selectedNodeId !== null) updateInspector(selectedNodeId);
}}

function resetView() {{
  if (is3D) {{
    cameraState = {{eye: {{x: 1.5, y: 1.5, z: 0.5}}, center: {{x:0, y:0, z:0}}, up: {{x:0, y:0, z:1}}}};
    Plotly.relayout('plotly-div', {{'scene.camera': cameraState}});
  }} else {{
    Plotly.relayout('plotly-div', {{
      'xaxis.autorange': true,
      'yaxis.autorange': true
    }});
  }}
}}

function toggleMode() {{
  is3D = document.getElementById('mode-toggle').checked;
  Plotly.newPlot('plotly-div', buildTraces(selectedNodeId), getLayout(), config);
  if (is3D) {{
    startAutoRotate();
  }} else {{
    stopAutoRotate();
  }}
}}

// Time slider
const twSlider = document.getElementById('tw-slider');
const twDisplay = document.getElementById('tw-display');

twSlider.addEventListener('input', function() {{
  currentTW = parseInt(this.value);
  if (currentTW < 0) {{
    twDisplay.textContent = 'All';
  }} else {{
    const twLabel = allTWs.includes(currentTW) ? points.find(p=>p.tw_idx===currentTW)?.tw_label || ('TW '+currentTW) : 'TW '+currentTW;
    twDisplay.textContent = twLabel;
  }}
  refresh();
}});

// Play/pause
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? '⏸' : '▶';
  if (playing) {{
    let idx = allTWs.indexOf(currentTW);
    if (idx < 0) idx = 0;
    playTimer = setInterval(() => {{
      idx = (idx + 1) % allTWs.length;
      currentTW = allTWs[idx];
      twSlider.value = currentTW;
      const twLabel = points.find(p=>p.tw_idx===currentTW)?.tw_label || ('TW '+currentTW);
      twDisplay.textContent = twLabel;
      refresh();
    }}, 1000);
  }} else {{
    clearInterval(playTimer);
  }}
}}

// Inspector
function updateInspector(nid) {{
  const idxs = nodeIndex[nid] || [];
  const p0 = idxs.length ? points[idxs[0]] : null;
  if (!p0) return;
  const tws = idxs.map(i => points[i]).sort((a,b) => a.tw_idx - b.tw_idx);

  let html = '<div class="card node-info">';
  html += `<button class="icon-btn" onclick="clearSelection()" style="float:right; padding:4px 8px; margin-top:-4px;">&times; Deselect</button>`;
  html += `<b>ID:</b> ${{nid}}<br>`;
  html += `<b>Type:</b> ${{p0.type}}<br>`;
  html += `<b>Path:</b> ${{p0.path}}<br>`;
  if (p0.cmd) html += `<b>Cmd:</b> ${{p0.cmd}}<br>`;
  html += `<b>Label:</b> ${{p0.label === 1 ? '<span style="color:'+getColor(p0)+'">Malicious</span>' : 'Benign'}}<br>`;
  if (p0.detection_status === 1) html += `<b>Detection:</b> <span style="color:${{getColor(p0)}}">Detected</span><br>`;
  else if (p0.detection_status === 2) html += `<b>Detection:</b> <span style="color:${{getColor(p0)}}">Undetected</span><br>`;
  html += `<b>Appearances:</b> ${{tws.length}} time windows<br>`;
  html += '</div>';

  html += `<div class="card"><h3>Causal Tracing</h3>`;
  html += `<p style="font-size:11px;color:#aaa;margin-bottom:8px">Isolate the temporal impact (forward) and origin (backward) of this node.</p>`;
  if (activeTraceNodes) {{
    html += `<button class="icon-btn" style="width:100%;background:rgba(255,80,80,0.2)" onclick="activeTraceNodes=null;refresh();updateInspector(${{nid}})">Clear Causal Trace</button>`;
    html += `<div style="margin-top:8px;font-size:12px;color:#64b4ff">Isolated ${{activeTraceNodes.size}} causally linked nodes.</div>`;
  }} else {{
    html += `<button class="icon-btn" style="width:100%;background:rgba(100,180,255,0.2)" onclick="performCausalTrace(${{nid}})">Trace Causal Chain</button>`;
  }}
  html += `</div>`;

  // Temporal trajectory with clickable TW buttons
  html += '<div class="card"><h3>Temporal Trajectory</h3>';
  html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">';
  html += `<button class="tw-btn ${{currentTW<0?'active':''}}" onclick="twSlider.value=-1;twSlider.dispatchEvent(new Event('input'))">All</button>`;
  tws.forEach(tw => {{
    html += `<button class="tw-btn ${{currentTW===tw.tw_idx?'active':''}}" onclick="twSlider.value=${{tw.tw_idx}};twSlider.dispatchEvent(new Event('input'))">TW ${{tw.tw_idx}}</button>`;
  }});
  html += '</div>';
  tws.forEach(tw => {{
    html += `<div class="trajectory-info">TW ${{tw.tw_idx}} &mdash; ${{tw.tw_label}} &bull; (${{tw.x.toFixed(3)}}, ${{tw.y.toFixed(3)}})</div>`;
  }});
  html += '</div>';

  // Neighbors
  const hops = parseInt(document.getElementById('hop-slider').value);
  if (hops > 0) {{
    const neighbors = getKHopNeighbors(nid, hops);
    html += '<div class="card"><h3>Neighbors (' + hops + '-hop, ' + neighbors.length + ' nodes)</h3>';
    neighbors.slice(0, 50).forEach(nbr => {{
      const nbr_pts = nodeIndex[nbr] || [];
      if (!nbr_pts.length) return;
      const np0 = points[nbr_pts[0]];
      html += `<div class="neighbor-item" style="border-left:3px solid ${{getColor(np0)}}" onclick="selectNode(${{nbr}})">${{np0.type}}: ${{np0.path.substring(0,30)}} (ID:${{nbr}})</div>`;
    }});
    if (neighbors.length > 50) html += `<div class="empty-state">... and ${{neighbors.length - 50}} more</div>`;
    html += '</div>';
  }}

  document.getElementById('inspector-content').innerHTML = html;
}}

function getKHopNeighbors(nid, k) {{
  const visited = new Set([nid]);
  let frontier = [nid];
  for (let h = 0; h < k; h++) {{
    const next = [];
    frontier.forEach(n => {{
      (adj[String(n)] || []).forEach(edge => {{
        if (!visited.has(edge.nb)) {{ visited.add(edge.nb); next.push(edge.nb); }}
      }});
    }});
    frontier = next;
  }}
  visited.delete(nid);
  return [...visited];
}}

function getCausalTrace(nid) {{
  const traceNodes = new Set([nid]);

  // Forward BFS
  let forwardQ = [];
  (adj[String(nid)] || []).forEach(edge => {{
    if (edge.dir === 'out') forwardQ.push({{node: edge.nb, time: edge.t}});
  }});
  const visitedFwd = new Set();

  while(forwardQ.length > 0) {{
    const curr = forwardQ.shift();
    const stateKey = `${{curr.node}}-${{curr.time}}`;
    if (visitedFwd.has(stateKey)) continue;
    visitedFwd.add(stateKey);
    traceNodes.add(curr.node);

    (adj[String(curr.node)] || []).forEach(edge => {{
      if (edge.dir === 'out' && edge.t >= curr.time) {{
        forwardQ.push({{node: edge.nb, time: edge.t}});
      }}
    }});
  }}

  // Backward BFS
  let backwardQ = [];
  (adj[String(nid)] || []).forEach(edge => {{
    if (edge.dir === 'in') backwardQ.push({{node: edge.nb, time: edge.t}});
  }});
  const visitedBwd = new Set();

  while(backwardQ.length > 0) {{
    const curr = backwardQ.shift();
    const stateKey = `${{curr.node}}-${{curr.time}}`;
    if (visitedBwd.has(stateKey)) continue;
    visitedBwd.add(stateKey);
    traceNodes.add(curr.node);

    (adj[String(curr.node)] || []).forEach(edge => {{
      if (edge.dir === 'in' && edge.t <= curr.time) {{
        backwardQ.push({{node: edge.nb, time: edge.t}});
      }}
    }});
  }}

  return traceNodes;
}}

function performCausalTrace(nid) {{
  activeTraceNodes = getCausalTrace(nid);
  refresh();
  updateInspector(nid);
}}

// Controls
document.getElementById('hop-slider').addEventListener('input', e => {{
  document.getElementById('hop-val').textContent = e.target.value;
  if (selectedNodeId !== null) updateInspector(selectedNodeId);
}});
['cb-benign','cb-detected','cb-undetected'].forEach(id => {{
  document.getElementById(id).addEventListener('change', refresh);
}});

// Search
document.getElementById('search-box').addEventListener('input', function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{ clearSelection(); return; }}
  const numId = parseInt(q);
  if (!isNaN(numId) && nodeIndex[numId]) {{ selectNode(numId); return; }}
  for (const p of points) {{
    if (p.path.toLowerCase().includes(q) || p.cmd.toLowerCase().includes(q)) {{
      selectNode(p.node_id); return;
    }}
  }}
}});

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') clearSelection(); }});

}} // end _initApp()
_loadData();
</script>
</body>
</html>"""
