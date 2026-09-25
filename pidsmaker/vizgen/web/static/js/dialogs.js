/* dialogs.js — causal subgraph, anomalous edges, score distribution */
const Dialogs = (() => {
  const $ = (id) => document.getElementById(id);
  const mount = () => $('modal_mount');

  function close() { mount().innerHTML = ''; }
  function shell(title, w) {
    mount().innerHTML =
      `<div class="modal-bg"><div class="modal" style="max-width:${w || 1100}px">` +
      `<h2>${title}</h2><div class="body" id="dlg_body"></div>` +
      `<div class="foot"><button id="dlg_close">Close</button></div></div></div>`;
    $('dlg_close').onclick = close;
    return $('dlg_body');
  }

  // Row metadata (type / score / path / label …) is enriched server-side from
  // the full dataset, so these tables are complete even for nodes outside the
  // client's loaded LOD prefix.
  const scoreColor = (s) => s > 0.5 ? '#ef4444' : s > 0.1 ? '#eab308' : '';

  // ----------------------- causal subgraph -----------------------
  async function causal() {
    if (App.S.selectedId === null) { alert('Select a node first.'); return; }
    if (!App.S.run.has_adj) { alert('No adjacency data for this run.'); return; }
    const start = App.S.selectedId;
    App.showLoading(true, 'Tracing causal subgraph…');
    let data;
    try { data = await Data.getCausal(App.S.run.file, start); }
    catch (e) { alert('Failed to trace causal subgraph: ' + e.message); return; }
    finally { App.showLoading(false); }
    const rows = data.rows.slice().sort((a, b) => (a.twi ?? 0) - (b.twi ?? 0));
    const rowMap = new Map(data.rows.map((r) => [r.id, r]));

    const body = shell(`Causal Subgraph for Node ${start}`, 1100);
    body.innerHTML =
      `<div style="color:#60a5fa;margin-bottom:6px">Traced ${data.count} causally linked nodes ` +
      `(events that could have influenced or been influenced by this node, ordered in time).</div>` +
      `<div class="tabbar"><span class="tab active" data-t="tbl">Chronological Table</span>` +
      `<span class="tab" data-t="g">Directed Graph View</span></div>` +
      `<input type="text" id="cz_search" placeholder="Filter by node ID, type, or path...">` +
      `<div id="cz_tbl" style="max-height:50vh;overflow:auto;margin-top:8px"></div>` +
      `<div id="cz_g" class="hidden"></div>`;
    renderCausalTable(rows, start);
    body.querySelectorAll('.tab').forEach((tab) => tab.onclick = () => {
      body.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const g = tab.dataset.t === 'g';
      $('cz_g').classList.toggle('hidden', !g);
      $('cz_tbl').classList.toggle('hidden', g);
      if (g && !$('cz_g').dataset.drawn) {
        renderCausalGraph(data.graph_nodes, data.graph_edges, start, rowMap);
        $('cz_g').dataset.drawn = '1';
      }
    });
    $('cz_search').addEventListener('input', (e) => renderCausalTable(rows, start, e.target.value.toLowerCase()));
  }

  function renderCausalTable(rows, start, filter) {
    let h = '<table class="data"><tr><th>Time Window</th><th>Node ID</th><th>Type</th><th>Score</th><th>Path / Cmd</th></tr>';
    rows.forEach((r) => {
      if (filter && !(`${r.id} ${r.type} ${r.path}`.toLowerCase().includes(filter))) return;
      const origin = r.id === start;
      const idc = r.label ? '#ef4444' : '#10b981';
      h += `<tr class="${origin ? 'origin' : ''}">` +
        `<td style="color:${idc}">${r.tw}</td>` +
        `<td style="color:${idc}">${r.id}${origin ? ' (Origin)' : ''}</td>` +
        `<td>${r.type}</td><td style="color:${scoreColor(r.score)}">${(r.score || 0).toFixed(4)}</td>` +
        `<td>${(r.cmd && r.cmd !== 'None') ? r.cmd : r.path}</td></tr>`;
    });
    $('cz_tbl').innerHTML = h + '</table>';
  }

  function renderCausalGraph(nodes, edges, start, rowMap) {
    // nodes = origin + malicious (server-capped); edges = [[src,dst]] among them
    const nset = new Set(nodes);
    if (nodes.length <= 1) {
      $('cz_g').innerHTML = '<div style="color:#ef4444;padding:30px;text-align:center">⚠️ Node is ISOLATED.<br>No causal edges in the raw dataset.</div>';
      return;
    }
    const out = new Map();
    edges.forEach(([s, t]) => { if (!out.has(s)) out.set(s, []); out.get(s).push(t); });
    // BFS depth from origin for x layout
    const depth = new Map([[start, 0]]); const q = [start];
    while (q.length) {
      const u = q.shift();
      for (const v of (out.get(u) || [])) {
        if (nset.has(v) && !depth.has(v)) { depth.set(v, depth.get(u) + 1); q.push(v); }
      }
    }
    const maxD = Math.max(1, ...[...depth.values()]);
    const byDepth = {}; nodes.forEach((id) => { const d = depth.get(id) ?? maxD; (byDepth[d] = byDepth[d] || []).push(id); });
    const W = 1000, Hh = 520, pos = new Map();
    Object.entries(byDepth).forEach(([d, arr]) => {
      arr.forEach((id, i) => pos.set(id, [60 + (W - 120) * (d / maxD), 30 + (Hh - 60) * ((i + 1) / (arr.length + 1))]));
    });
    let svg = `<svg width="100%" viewBox="0 0 ${W} ${Hh}" style="background:#111115;border-radius:6px">`;
    // edges
    edges.forEach(([u, v]) => {
      if (pos.has(u) && pos.has(v)) {
        const a = pos.get(u), b = pos.get(v);
        svg += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#4a4a5b" stroke-width="0.8" opacity="0.5"/>`;
      }
    });
    // nodes
    nodes.forEach((id) => {
      const p = pos.get(id); if (!p) return;
      const r = rowMap.get(id) || { label: 0, score: 0, type: '', path: '' };
      const col = id === start ? '#f59e0b' : (r.label ? '#ef4444' : '#10b981');
      const sz = Math.max(4, Math.min(11, 4 + (r.score || 0) * 1.5));
      svg += `<circle cx="${p[0]}" cy="${p[1]}" r="${sz}" fill="${col}"><title>ID: ${id}\nType: ${r.type}\nScore: ${(r.score || 0).toFixed(4)}\nPath: ${r.path}</title></circle>`;
      if (nodes.length < 40 || id === start) {
        const lab = (r.path || String(id)).split('/').pop().slice(0, 12);
        svg += `<text x="${p[0] + sz + 2}" y="${p[1] + 3}" fill="#a0a0b0" font-size="9">${lab}</text>`;
      }
    });
    svg += '</svg>';
    $('cz_g').innerHTML = svg;
  }

  // ----------------------- anomalous edges -----------------------
  async function neighbors() {
    if (App.S.selectedId === null) { alert('Select a node first.'); return; }
    if (!App.S.run.has_adj) { alert('No adjacency data for this run.'); return; }
    const center = App.S.selectedId;
    App.showLoading(true, 'Loading edges…');
    let nbrs;
    try { nbrs = (await Data.getNeighbors(App.S.run.file, center)).edges; }
    catch (e) { alert('Failed to load edges: ' + e.message); return; }
    finally { App.showLoading(false); }
    const hasEt = nbrs.some((e) => e.et);  // edge types present in this run's data?

    const body = shell(`Anomalous Edges for Node ${center}`, 1040);
    body.innerHTML =
      '<div style="display:flex;gap:16px;align-items:center;margin-bottom:8px">' +
      '<label>Direction: <select id="ne_dir"><option value="all">All</option>' +
      '<option value="in">in</option><option value="out">out</option></select></label>' +
      '<label class="chk2"><input type="checkbox" id="ne_group" checked> ' +
      'Group identical edges across time windows</label>' +
      (hasEt ? '' : '<span class="muted" style="font-size:11px">(edge types unavailable for this run — regenerate to populate)</span>') +
      '<button id="ne_csv" style="margin-left:auto">Export CSV</button>' +
      '</div><div id="ne_info" style="color:#60a5fa;margin-bottom:6px"></div>' +
      '<div id="ne_tbl" style="max-height:55vh;overflow:auto"></div>';

    let lastRows = [], lastGrouped = true;
    function render() {
      const dirF = $('ne_dir').value, grp = $('ne_group').checked;
      const sel = dirF === 'all' ? nbrs : nbrs.filter((e) => e.dir === dirF);
      let rows;
      // Time window of the edge = the neighbour's time window (twi). The raw edge
      // time (e.t) is not populated by generation (the source graphs carry no
      // edge_time), so it was always 0 — use the neighbour's real tw instead.
      const twOf = (e) => (e.twi != null ? e.twi : e.t);
      if (grp) {
        // collapse the same (neighbor, dir, edge-type) seen across time windows;
        // per-neighbor metadata (type/score/path…) is carried on each edge.
        const g = new Map();
        sel.forEach((e) => {
          const et = e.et || '';
          const k = e.nb + '|' + e.dir + '|' + et;
          if (!g.has(k)) g.set(k, {
            nb: e.nb, dir: e.dir, et, tws: [],
            type: e.type, score: e.score, path: e.path, cmd: e.cmd, label: e.label,
          });
          const tw = twOf(e);
          if (!g.get(k).tws.includes(tw)) g.get(k).tws.push(tw);
        });
        rows = [...g.values()];
      } else {
        rows = sel.map((e) => ({
          nb: e.nb, dir: e.dir, et: e.et || '', tws: [twOf(e)],
          type: e.type, score: e.score, path: e.path, cmd: e.cmd, label: e.label,
        }));
      }
      rows.sort((a, b) => (b.score || 0) - (a.score || 0));
      lastRows = rows; lastGrouped = grp;
      $('ne_info').textContent =
        `Showing ${rows.length} ${grp ? 'unique edges' : 'edge instances'} for Node ${center}, sorted by anomaly score.`;
      let h = '<table class="data"><tr><th>Dir</th><th>Edge Type</th><th>Time Window' +
        (grp ? 's' : '') + '</th><th>Node ID</th><th>Type</th><th>Score</th><th>Path / Cmd</th></tr>';
      rows.forEach((r) => {
        const idc = r.label ? '#ef4444' : '#10b981';
        const sorted = grp ? r.tws.slice().sort((a, b) => a - b) : r.tws;
        const tws = grp
          ? (sorted.slice(0, 2).join(', ') + (sorted.length > 3 ? ` (+${sorted.length - 2} more)` : ''))
          : sorted[0];
        h += `<tr><td>${r.dir}</td><td>${r.et || '—'}</td><td style="color:${idc}">${tws}</td>` +
          `<td style="color:${idc}">${r.nb}</td><td>${r.type}</td>` +
          `<td style="color:${scoreColor(r.score)}">${(r.score || 0).toFixed(4)}</td>` +
          `<td>${(r.cmd && r.cmd !== 'None') ? r.cmd : r.path}</td></tr>`;
      });
      $('ne_tbl').innerHTML = h + '</table>';
    }
    $('ne_dir').addEventListener('change', render);
    $('ne_group').addEventListener('change', render);
    $('ne_csv').addEventListener('click', () => {
      const q = (v) => { const s = String(v ?? ''); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
      const head = ['Dir', 'Edge Type', lastGrouped ? 'Time Windows' : 'Time Window',
        'Node ID', 'Type', 'Score', 'Path / Cmd'];
      const lines = [head.join(',')];
      lastRows.forEach((r) => {
        const tws = lastGrouped ? r.tws.join(' ') : r.tws[0];
        const pc = (r.cmd && r.cmd !== 'None') ? r.cmd : r.path;
        lines.push([r.dir, r.et || '', tws, r.nb, r.type, (r.score || 0).toFixed(4), pc].map(q).join(','));
      });
      const f = App.S.run ? App.S.run.file : '';
      const m = f.match(/_epoch_(\d+)/);
      const best = (App.S.run && App.S.run.epochs || []).find((e) => e.is_best);
      const ep = m ? m[1] : (best && best.epoch != null ? best.epoch : 'NA');
      const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `edges_node_${center}_epoch_${ep}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
    render();
  }

  // ----------------------- score distribution -----------------------
  // Rendered server-side with the exact native matplotlib code, served as a
  // PNG so it is pixel-identical to the desktop viewer.
  function scoreDist() {
    if (!App.S.run) return;
    // same continuously-advancing loader bar as See Edges / Causal while the
    // server renders the plot; the modal opens once the image is ready.
    App.showLoading(true, 'Rendering score distribution…');
    const img = new Image();
    img.onload = () => {
      App.showLoading(false);
      const body = shell('Anomaly Score Distribution', 1000);
      body.innerHTML = '';
      img.style.width = '100%'; img.style.background = '#fff'; img.style.borderRadius = '4px';
      body.appendChild(img);
    };
    img.onerror = () => { App.showLoading(false); alert('Failed to render score distribution.'); };
    img.src = '/api/scoredist?file=' + encodeURIComponent(App.S.run.file) + '&t=' + Date.now();
  }

  return { causal, neighbors, scoreDist };
})();
