/* app.js — application state, render-state composition, interactions */
const App = (() => {
  const S = {
    run: null, dec: null, attackPairs: null,
    n: 0, baseColor: null, baseSize: null,
    displayColor: null, displaySize: null, visible: null,
    idToIdx: null, maxScore: 1,
    selectedId: null, selectedIdxs: [],
    searchRows: null, csvRows: null,   // server-side filter results (buffer-row sets)
    playTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  // Max nodes kept resident. A run bigger than this loads its first LOD_CAP nodes
  // (the exporter's natural order); the cap is the browser/GPU limit, not the
  // server. Buffer streams in LOD_CHUNK-sized steps that each render on arrival.
  const LOD_CAP = 25000000;   // load the whole run (covers 20M+); browser/GPU is the real limit
  // Bumped on every loadRun so a superseded load (user switched runs, or the
  // page navigated) bails quietly instead of popping a "Failed to load" alert
  // when its now-orphaned fetch is aborted.
  let loadToken = 0;
  let selToken = 0;      // bumped per selection so a stale on-demand meta fetch is dropped
  let searchToken = 0;   // bumped per search so a stale server-search result is dropped
  const PLASMA = ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921']
    .map((h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255]);

  function plasma(t) {
    t = Math.max(0, Math.min(1, t)) * (PLASMA.length - 1);
    const i = Math.floor(t), f = t - i, a = PLASMA[i], b = PLASMA[Math.min(i + 1, PLASMA.length - 1)];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  // Benign alphas run a bit higher than the old translucent look: with depthWrite
  // on you see ~one disc layer (nearer points occlude farther) instead of many
  // summed, so each point needs more opacity to keep clusters reading bright.
  const BENIGN = { 0: [0.8, 0.5, 1.0, 0.55], 1: [0.3, 1.0, 0.7, 0.52], 2: [0.5, 0.8, 1.0, 0.6], 3: [0.5, 0.5, 0.5, 0.18] };
  const DET_RED = [1.0, 0.2, 0.2, 0.9], UNDET_ORANGE = [1.0, 0.6, 0.2, 0.9];
  function nodeRgba(label, det, type) {
    if (label === 0) return BENIGN[type] || BENIGN[3];
    return det === 2 ? UNDET_ORANGE : DET_RED;
  }
  // type_enum → readable name (matches viz_server _type_enum: 0 proc,1 file,2 net,3 other)
  const TYPE_NAMES = ['process', 'file', 'netflow', 'other'];
  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: return [v, t, p]; case 1: return [q, v, p]; case 2: return [p, v, t];
      case 3: return [p, q, v]; case 4: return [t, p, v]; default: return [v, p, q];
    }
  }

  // ---- flags accessors ----
  // 0 id, 1 label, 2 det, 3 type. id is a u32 column; label/det/type are bit-packed
  // into one byte per node (label<<0 | det<<1 | type<<3).
  const flag = (i, k) => {
    if (k === 0) return S.dec.ids[i];
    const m = S.dec.meta[i];
    return k === 1 ? (m & 1) : k === 2 ? ((m >> 1) & 3) : ((m >> 3) & 3);
  };
  const attr = (i, k) => S.dec.attrs[i * 5 + k];   // 0 tw,1 start,2 end,3 score,4 size

  // First streaming step is small for a fast first paint; the step then doubles
  // up to LOD_CHUNK_MAX so a big run needs far fewer round trips over the tunnel
  // (20M ≈ 12 chunks instead of 80). The eased progress bar hides the step sizes.
  const LOD_CHUNK = 250000;        // first (smallest) streaming step
  const LOD_CHUNK_MAX = 3000000;   // cap on the growing step

  function setTotalLabel(loaded, n) {
    $('lbl_total').textContent = (loaded < n)
      ? `Showing ${loaded.toLocaleString()} of ${n.toLocaleString()} (sample)`
      : 'Total Nodes: ' + n;
  }

  // ---- smooth loader progress engine ----------------------------------------
  // One rAF loop eases the displayed fill (and the count) toward a moving
  // "ceiling" every frame, so the bar and the "N / total" number climb
  // continuously in real time instead of snapping in 500K jumps.
  //   determinate (run streaming): loaderProgress() raises the ceiling to the
  //     real fraction as each chunk arrives; the fill keeps sliding toward it,
  //     and the shown count = round(fill * cap) so it ticks up smoothly.
  //   creep (single dialog fetch — See Edges / Causal, no measurable progress):
  //     the ceiling drifts toward ~90% on its own so the bar always advances.
  const _pg = { raf: 0, cur: 0, ceil: 0, creep: false, cap: 0, total: 0 };
  function _pgTick() {
    if (_pg.creep) _pg.ceil = Math.min(0.9, _pg.ceil + (0.9 - _pg.ceil) * 0.035);
    _pg.cur += (_pg.ceil - _pg.cur) * 0.14;
    const f = Math.min(1, _pg.cur), pct = Math.round(f * 100);
    $('load_fill').style.transform = 'scaleX(' + f.toFixed(4) + ')';
    if (_pg.cap) {
      const shown = Math.min(_pg.cap, Math.round(f * _pg.cap));
      $('load_pct').textContent = `${shown.toLocaleString()} / ${_pg.cap.toLocaleString()}`
        + (_pg.cap < _pg.total ? ` (sample of ${_pg.total.toLocaleString()})` : '') + `  ·  ${pct}%`;
    } else {
      $('load_pct').textContent = `${pct}%`;
    }
    const settled = !_pg.creep && _pg.ceil >= 1 && _pg.cur > 0.999;
    _pg.raf = settled ? 0 : requestAnimationFrame(_pgTick);
  }
  function _pgKick() { if (!_pg.raf) _pg.raf = requestAnimationFrame(_pgTick); }
  function _pgBegin(creep) { _pg.cur = 0; _pg.ceil = creep ? 0.06 : 0; _pg.creep = !!creep; _pg.cap = 0; _pg.total = 0; _pgKick(); }
  function _pgStop() { if (_pg.raf) cancelAnimationFrame(_pg.raf); _pg.raf = 0; }
  // raise the ceiling to the real fraction; the fill + count slide toward it
  function loaderProgress(loaded, cap, total) {
    _pg.creep = false; _pg.cap = cap; _pg.total = total;
    _pg.ceil = Math.min(1, Math.max(_pg.ceil, cap ? loaded / cap : 0));
    _pgKick();
  }
  // hold at 100% just long enough for the fill + count to visibly complete
  function loaderFinish() { _pg.creep = false; _pg.ceil = 1; _pgKick(); return new Promise((r) => setTimeout(r, 260)); }

  // #loading = message + bar + %. `bar`: true → determinate (real count);
  // 'creep' → auto-advancing bar for single fetches; falsy → message only.
  function loaderShow(on, msg, bar) {
    $('loading').style.display = on ? 'flex' : 'none';
    if (msg) $('load_msg').textContent = msg;
    const showBar = on && !!bar;
    $('load_bar').style.display = showBar ? '' : 'none';
    $('load_fill').style.display = showBar ? '' : 'none';
    $('load_pct').style.display = showBar ? '' : 'none';
    if (showBar) { $('load_fill').style.transform = 'scaleX(0)'; _pgBegin(bar !== true); }
    else _pgStop();
  }
  const nextFrame = () => new Promise((r) => requestAnimationFrame(r));
  // dialog fetches (See Edges, Extract Causal): a continuously-advancing bar + %
  function showLoading(on, msg) { loaderShow(on, msg, 'creep'); }

  function populateRunUI(run) {
    $('hops').max = run.hops - 1; $('hops').value = run.hops - 1;
    $('lbl_hops').textContent = `Hops (${run.hops - 1}):`;
    $('row_hops').style.display = run.hops > 1 ? '' : 'none';
    $('slider_tw').max = (run.max_tw || 0) * 100;
    $('slider_tw').value = -100; $('lbl_tw').textContent = 'All';
    UI.populateEpochs(run);
    UI.fillStats(run); UI.fillDetection(run.detection_cost);
    $('txt_config').textContent = run.run_config || '# No run_config.yml found for this run.';
    $('lbl_model').textContent = 'Model: ' + run.model;
    // the embedding-space picker only makes sense when both spaces were exported
    $('emb_wrap').classList.toggle('hidden', !(run.word2vec_file && run.encoder_file));
    // any file that is not the featurization projection is an encoder projection
    // (a bare encoder file, or one of its per-epoch files)
    $('cmb_emb').value = run.file === run.word2vec_file ? 'feat' : 'enc';
  }

  // TIER 1 app state for a streamed core chunk: just colour (from label/det/type
  // in the meta byte) and size (by label). Kept deliberately cheap so the cloud
  // paints fast. The id→instances map is built in tier 2 — it is only needed by
  // the temporal/attack overlays, and at 20M it is the single most expensive
  // structure to build, so it must not gate the first paint. (String metadata is
  // fetched per node on demand; it is never bulk-loaded.)
  function appendCoreState(off, len) {
    for (let i = 0; i < len; i++) {
      const g = off + i;
      S.baseColor.set(nodeRgba(flag(g, 1), flag(g, 2), flag(g, 3)), g * 4);
      S.baseSize[g] = flag(g, 1) !== 0 ? 5.0 : 3.0;   // malicious bigger; matches attrs[4]
    }
  }

  // TIER 2 app state for a streamed aux chunk: the max anomaly score (heatmap
  // normalisation) and the id→instances map (temporal trajectories / attack
  // overlay). tw start/end are handled in the scene.
  //
  // This is the heaviest background step — millions of Map insertions + array
  // pushes, which used to run as one synchronous stint per (growing, up to 3M-row)
  // chunk and froze the UI for a beat right after the cloud appeared. It now runs
  // in small time-budgeted slices: process rows until ~AUX_BUDGET_MS have elapsed
  // this frame, then yield so the renderer paints, then resume. The main thread
  // stays at ~60fps throughout — the work is spread across frames, not blocked in
  // one lump. A superseded load (newer token) bails mid-way.
  const AUX_BUDGET_MS = 8;     // main-thread work per frame before yielding
  const AUX_CLOCK_STEP = 8192; // rows between wall-clock checks (checking every row is itself costly)
  async function appendAuxState(off, len, token) {
    let i = 0;
    while (i < len) {
      const t0 = performance.now();
      do {
        const end = Math.min(i + AUX_CLOCK_STEP, len);
        for (; i < end; i++) {
          const g = off + i;
          const sc = attr(g, 3);
          if (sc > S.maxScore) S.maxScore = sc;
          const id = flag(g, 0);
          let arr = S.idToIdx.get(id);
          if (!arr) { arr = []; S.idToIdx.set(id, arr); }
          arr.push(g);
        }
      } while (i < len && performance.now() - t0 < AUX_BUDGET_MS);
      if (token !== loadToken) return;
      await nextFrame();
    }
  }

  // Render just the newly-streamed rows [off, off+len): base colours/sizes + the
  // label/search/CSV visibility, uploaded as one slice. Overlays (heatmap/FP) and
  // selection are applied once by the full refresh() at the end of the load — so
  // streaming is O(total), not a full-cloud recompute per chunk (O(n²)).
  function renderChunk(off, len) {
    S.displayColor.set(S.baseColor.subarray(off * 4, (off + len) * 4), off * 4);
    S.displaySize.set(S.baseSize.subarray(off, off + len), off);
    const fb = $('chk_benign').checked, fd = $('chk_det').checked, fu = $('chk_undet').checked;
    const sr = S.searchRows, cr = S.csvRows;   // buffer-row sets (or null)
    for (let i = off; i < off + len; i++) {
      const lbl = flag(i, 1), det = flag(i, 2);
      let show = (lbl === 0 && fb) || (lbl !== 0 && (det === 0 || det === 1) && fd) || (lbl !== 0 && det === 2 && fu);
      if (show && (sr || cr)) show = (!sr || sr.has(i)) && (!cr || cr.has(i));
      S.visible[i] = show ? 1 : 0;
    }
    Scene.updateDisplayRange(off, len, S.displayColor, S.displaySize, S.visible);
  }

  async function loadRun(file) {
    const token = ++loadToken;
    loaderShow(true, 'Loading run…', true);
    try {
      const run = await Data.getRun(file);
      if (token !== loadToken) return;
      const cap = Math.min(run.n, LOD_CAP);

      // allocate accumulators sized to cap; the scene accumulator is shared as S.dec.
      S.run = run; S.attackPairs = null; S.n = 0; S.maxScore = 1e-9;
      S.selectedId = null; S.selectedIdxs = [];
      S.searchRows = null; S.csvRows = null;   // server-side filter results (buffer-row sets)
      S.auxReady = false;                    // tier-2 (time/score) attrs not loaded yet
      S.baseColor = new Float32Array(cap * 4); S.baseSize = new Float32Array(cap);
      S.displayColor = new Float32Array(cap * 4); S.displaySize = new Float32Array(cap);
      S.visible = new Float32Array(cap);
      S.idToIdx = new Map();
      Scene.beginLoad(cap, run.hops, true);
      S.dec = Scene.decoded;
      populateRunUI(run);
      clearSelection();

      // TIER 1 — stream positions + colours in chunks; the cloud renders as it
      // arrives. Chunk size GROWS (small first chunk → fast first paint; then
      // bigger chunks → far fewer round trips over the tunnel for a big run).
      for (let off = 0, step = LOD_CHUNK; off < cap; off += step, step = Math.min(step * 2, LOD_CHUNK_MAX)) {
        const len = Math.min(step, cap - off);
        // aim the bar/count at this chunk's end BEFORE fetching, so the fill and
        // the "N / total" number climb continuously while the chunk downloads.
        loaderProgress(off + len, cap, run.n);
        const dec = await Data.getBufferCore(run, off, len);
        if (token !== loadToken) return;            // superseded by a newer load
        Scene.appendCore(dec, off);
        appendCoreState(off, len);
        S.n = off + len;
        renderChunk(off, len);   // upload only this slice (not a full O(n) refresh)
        await nextFrame();  // let the browser paint the new chunk
      }
      await loaderFinish();  // let the fill + count visibly reach 100%
      loaderShow(false);
      setTotalLabel(cap, run.n);
      Scene.refreshBounds();   // one full-cloud bbox pass now that everything is in
      refresh();               // one full render pass
      try { localStorage.setItem('pids_last_run', file); } catch (e) { }
      // TIER 2 — the cloud is up; pull the time/score attrs in the background so
      // the slider, playback, heatmap, FP and attack-graph overlays light up.
      loadAux(run, cap, token);
    } catch (e) {
      if (token === loadToken) { alert('Failed to load run: ' + e.message); loaderShow(false); }
    }
  }

  // TIER 2 background stream: the time/score attrs. Runs after the cloud is on
  // screen; a superseded load (newer token) bails, and a failure just leaves the
  // temporal/overlay features unavailable (the cloud stays fully usable).
  async function loadAux(run, cap, token) {
    // Fixed, moderate chunk (not the growing tier-1 step): keeps each decode /
    // scene-scatter stint well under a frame, and pipelines the NEXT chunk's fetch
    // so the network round-trip overlaps the current chunk's main-thread apply
    // (fetch and CPU run in parallel) instead of stalling between chunks.
    const AUX_CHUNK = 500000;
    const fetchAt = (o) => (o < cap ? Data.getBufferAux(run, o, Math.min(AUX_CHUNK, cap - o)).catch(() => null) : null);
    let pending = fetchAt(0);
    for (let off = 0; off < cap; off += AUX_CHUNK) {
      const len = Math.min(AUX_CHUNK, cap - off);
      const dec = await pending;
      if (token !== loadToken) return;
      if (!dec) return;                       // fetch failed → leave overlays unavailable
      pending = fetchAt(off + len);           // kick the next fetch off NOW (overlaps the apply below)
      Scene.appendAux(dec, off);
      await appendAuxState(off, len, token);  // time-sliced; yields each frame
    }
    if (token !== loadToken) return;
    S.auxReady = true;
    refresh();   // apply whatever temporal/overlay state is set, now that attrs exist
  }

  // ---- visibility (filters + search + csv) ----
  // Fill `out[0..S.n)` with the label/detection + search/CSV visibility. Search and
  // CSV are resolved server-side into buffer-ROW sets (row == buffer index), so
  // filtering is a direct index membership test — no ids needed. `hideSelected`
  // also hides the selected instances (they are drawn by the highlight layer);
  // picking passes false so it can still hit them.
  // Snapshot the active filter checkboxes/row-sets once, so the per-point
  // predicate below can be reused both for the full O(n) pass and for the tiny
  // incremental update on selection (a few indices) without re-reading the DOM.
  function visFilters() {
    return {
      fb: $('chk_benign').checked, fd: $('chk_det').checked, fu: $('chk_undet').checked,
      sr: S.searchRows, cr: S.csvRows,
    };
  }
  function visOne(i, f) {
    const lbl = flag(i, 1), det = flag(i, 2);
    let show = (lbl === 0 && f.fb) || (lbl !== 0 && (det === 0 || det === 1) && f.fd) || (lbl !== 0 && det === 2 && f.fu);
    if (show && (f.sr || f.cr)) show = (!f.sr || f.sr.has(i)) && (!f.cr || f.cr.has(i));
    return show ? 1 : 0;
  }
  function computeVisibleInto(out, hideSelected) {
    const f = visFilters();
    for (let i = 0; i < S.n; i++) out[i] = visOne(i, f);
    if (hideSelected && S.selectedIdxs.length) for (const i of S.selectedIdxs) out[i] = 0;
  }
  function computeVisible() { computeVisibleInto(S.visible, true); }

  // Invariant kept by both refresh() and refreshSelection(): S.visible holds the
  // filter visibility with the currently selected instances forced hidden (they
  // are drawn by the highlight layer instead). On a selection change we only need
  // to restore the previously hidden instances and hide the new ones — a handful
  // of writes plus one aVisible upload, versus a full recompute + full 4·n colour
  // re-upload. Colours/sizes never depend on selection, so they are left alone.
  function refreshSelection(prevIdxs) {
    if (!S.run || !S.visible) return;
    const f = visFilters();
    const touched = [];
    if (prevIdxs) for (const i of prevIdxs) { S.visible[i] = visOne(i, f); touched.push(i); }
    for (const i of S.selectedIdxs) { S.visible[i] = 0; touched.push(i); }
    Scene.updateVisibleAt(S.visible, touched);   // upload only the rows that changed
    Scene.setTime((S.auxReady && $('chk_temporal').checked) ? tval() : -1);
    Scene.setDim(S.selectedIdxs.length > 0);
    if (S.selectedIdxs.length) {
      const hp = [];
      for (const i of S.selectedIdxs) { const p = Scene.nodePos(i); hp.push(p[0], p[1], p[2]); }
      Scene.setHighlight(hp);
    } else Scene.setHighlight(null);
    computeTrajectory();
    computeAttack();
  }

  // ---- display color/size (overlays) ----
  function computeDisplay() {
    S.displayColor.set(S.baseColor);
    S.displaySize.set(S.baseSize);
    const dc = S.run.detection_cost;
    // heatmap / false-positive overlays read the per-node score (tier-2 attrs);
    // until those load, only the base colours are shown.
    if (!S.auxReady) return;
    const fpCamp = $('chk_fp_campaign').checked, fpRec = $('chk_fp_recall').checked;
    const heat = $('chk_heat').checked;

    if ((fpCamp || fpRec) && dc) {
      const thr = fpRec ? dc.thresh_full_recall : dc.thresh_full_campaign;
      if (thr !== null && thr !== undefined) {
        let fpCount = 0;
        for (let i = 0; i < S.n; i++) if (flag(i, 1) === 0 && attr(i, 3) >= thr) fpCount++;
        const fpAlpha = fpCount > 1000 ? 0.3 : 0.8, fpSize = fpCount > 1000 ? 4.0 : 7.0;
        for (let i = 0; i < S.n; i++) {
          const lbl = flag(i, 1), pos = attr(i, 3) >= thr, o = i * 4;
          let col, sz;
          if (lbl === 0 && !pos) { col = [0.2, 0.2, 0.2, 0.1]; sz = 2.0; }
          else if (lbl !== 0 && !pos) { col = [1.0, 0.2, 0.2, 0.1]; sz = 2.0; }
          else if (lbl !== 0 && pos) { col = [1.0, 0.2, 0.2, 0.4]; sz = 5.0; }
          else { col = [1.0, 0.7, 0.0, fpAlpha]; sz = fpSize; }
          S.displayColor[o] = col[0]; S.displayColor[o + 1] = col[1];
          S.displayColor[o + 2] = col[2]; S.displayColor[o + 3] = col[3];
          S.displaySize[i] = sz;
        }
      }
    } else if (heat) {
      for (let i = 0; i < S.n; i++) {
        if (flag(i, 1) !== 0) continue; // benign only
        const c = plasma(attr(i, 3) / S.maxScore), o = i * 4;
        S.displayColor[o] = c[0]; S.displayColor[o + 1] = c[1]; S.displayColor[o + 2] = c[2];
      }
    }
    // 2D flatten dims benign
    if (!$('chk_temporal').checked) {
      for (let i = 0; i < S.n; i++) if (flag(i, 1) === 0) S.displayColor[i * 4 + 3] *= 0.55;
    }
  }

  // ---- trajectory + attack edges (time-aware) ----
  function tval() {
    const v = +$('slider_tw').value;
    return v < 0 ? -1 : v / 100;
  }

  function computeTrajectory() {
    // trajectories order instances by time window (tier-2 attr)
    if (!S.auxReady || !$('chk_traj').checked || S.selectedId === null || S.selectedIdxs.length < 2) {
      Scene.setTrajectory(null); return;
    }
    const t = tval();
    const seq = S.selectedIdxs.map((i) => ({ tw: attr(i, 0), i })).sort((a, b) => a.tw - b.tw);
    const pts = [], cols = [];
    const n = seq.length;
    for (let k = 0; k < n; k++) {
      if (t >= 0 && seq[k].tw > t) break;
      const p = Scene.nodePos(seq[k].i);
      const ratio = k / Math.max(n - 1, 1);
      const c = hsv2rgb(0.65 - ratio * 0.65, 1, 1);
      const op = 0.2 + 0.8 * Math.pow(ratio, 3);
      pts.push(p[0], p[1], p[2]); cols.push(c[0], c[1], c[2], op);
    }
    Scene.setTrajectory(pts, cols);
  }

  async function computeAttack() {
    // attack edges are activated by node time windows (tier-2 attr)
    if (!S.auxReady || !$('chk_attack').checked) { Scene.setAttackEdges(null); return; }
    const pairs = await ensureAttackPairs();
    if (!pairs) return;
    const t = tval(), tAct = t < 0 ? Infinity : t;
    const pts = [], cols = [];
    for (const [u, v] of pairs) {
      const iu = pickInstance(u, tAct), iv = pickInstance(v, tAct);
      if (iu < 0 || iv < 0) continue;
      const uf = firstTw(u), vf = firstTw(v), activation = Math.max(uf, vf);
      let col;
      if (tAct < activation) col = [1.0, 0.8, 0.0, 0.30];
      else if (tAct < activation + 4.0) col = [0.93, 0.26, 0.26, 0.85];
      else col = [0.93, 0.26, 0.26, 0.65];
      const pu = Scene.nodePos(iu), pv = Scene.nodePos(iv);
      pts.push(pu[0], pu[1], pu[2], pv[0], pv[1], pv[2]);
      cols.push(...col, ...col);
    }
    Scene.setAttackEdges(pts, cols);
  }

  function firstTw(id) {
    const idxs = S.idToIdx.get(id); if (!idxs) return Infinity;
    let m = Infinity; for (const i of idxs) m = Math.min(m, attr(i, 0)); return m;
  }
  function pickInstance(id, t) {
    const idxs = S.idToIdx.get(id); if (!idxs) return -1;
    let best = -1, bestTw = -Infinity;
    for (const i of idxs) {
      const tw = attr(i, 0);
      if ((t === Infinity || tw <= t) && tw >= bestTw) { bestTw = tw; best = i; }
    }
    return best >= 0 ? best : idxs[idxs.length - 1];
  }

  async function ensureAttackPairs() {
    if (S.attackPairs) return S.attackPairs;
    if (!S.run.has_adj) return null;
    showLoading(true, 'Loading attack edges…');
    try { S.attackPairs = (await Data.getAttackPairs(S.run.file)).pairs; }
    catch (e) { S.attackPairs = null; }
    finally { showLoading(false); }
    return S.attackPairs;
  }

  // ---- master refresh (apply_visual_state) ----
  function refresh() {
    if (!S.run) return;
    computeVisible();
    computeDisplay();
    Scene.updateDisplay(S.displayColor, S.displaySize, S.visible);
    // temporal filtering needs the tier-2 tw attrs; until they load, show all time
    Scene.setTime((S.auxReady && $('chk_temporal').checked) ? tval() : -1);
    Scene.setDim(S.selectedIdxs.length > 0);
    // highlight selected instances
    if (S.selectedIdxs.length) {
      const hp = [];
      for (const i of S.selectedIdxs) { const p = Scene.nodePos(i); hp.push(p[0], p[1], p[2]); }
      Scene.setHighlight(hp);
    } else Scene.setHighlight(null);
    computeTrajectory();
    computeAttack();
  }

  // ---- selection / inspector ----
  // Selecting a node fetches ONLY that node's (and its 20 KNN neighbours') string
  // metadata on demand, an O(1) server-side row lookup (MetaStore.node_at), so
  // it is instant even at 20M nodes and no bulk metadata is ever downloaded.
  function selectNode(idx) {
    // Selection is keyed on the clicked buffer index (works before ids load). The
    // node id + metadata come from /api/node?idxs=…; once we know the id (and if
    // the id→instances map is ready) we expand to all its time-window instances.
    const prev = S.selectedIdxs;
    S.selectedIdxs = [idx];
    S.selectedId = null;
    const k = knn(idx);
    const tok = ++selToken;
    refreshSelection(prev);             // highlight the clicked instance immediately
    UI.showInspector(idx, k, null);     // buffer-only fields now; strings fill in below
    Data.getNode(S.run.file, [idx, ...k.nbIdxs]).then(({ rows }) => {
      if (tok !== selToken) return;     // selection changed while fetching
      const self = rows[idx];
      const prevSel = S.selectedIdxs;
      if (self) {
        S.selectedId = self.id;
        S.selectedIdxs = (S.auxReady && S.idToIdx.get(self.id)) || [idx];
      }
      const prefixes = {};
      for (const j of k.nbIdxs) {
        const p = ((rows[j] && rows[j].path) || '').slice(0, 50);
        prefixes[p] = (prefixes[p] || 0) + 1;
      }
      const topPre = Object.entries(prefixes).sort((a, b) => b[1] - a[1]).slice(0, 3);
      UI.showInspector(idx, Object.assign({}, k, { topPre }), self || null);
      refreshSelection(prevSel);        // re-highlight now the id + instances are known
    }).catch(() => { /* buffer-only inspector already shown */ });
  }
  function clearSelection() {
    S.selectedId = null; S.selectedIdxs = [];
    $('info_lbl').textContent = 'Click a point to inspect...';
  }

  function knn(idx, K = 20) {
    // Nearest neighbours come from Scene's spatial grid (O(1)-ish) instead of an
    // O(n) scan of every point — that scan was the main per-click freeze at scale.
    const nbIdxs = Scene.knnIndices(idx, K);
    // Aggregate the neighbourhood from the BUFFER only (label/det/type-enum/score);
    // node types come from the type-enum, no metadata needed. nbIdxs is returned so
    // the caller can fetch just these neighbours' paths on demand for "top paths".
    let nb = 0, nd = 0, nu = 0; const types = {}; const scores = [];
    for (const i of nbIdxs) {
      const lbl = flag(i, 1), det = flag(i, 2);
      if (lbl === 0) nb++; else if (det === 2) nu++; else nd++;
      const tp = TYPE_NAMES[flag(i, 3)] || 'other';
      types[tp] = (types[tp] || 0) + 1;
      scores.push(attr(i, 3));
    }
    scores.sort((a, b) => a - b);
    const med = scores.length ? scores[Math.floor(scores.length / 2)] : 0;
    const purity = Math.max(nb, nd, nu) / Math.max(nbIdxs.length, 1);
    const gap = attr(idx, 3) - med;
    return { nb, nd, nu, types, purity, gap, K: nbIdxs.length, nbIdxs };
  }

  // ---- playback ----
  function togglePlay() {
    if (!S.auxReady) return;   // temporal data (tier-2) still loading
    if (S.playTimer) { stopPlay(); return; }
    const max = +$('slider_tw').max;
    if (+$('slider_tw').value >= max) $('slider_tw').value = -100;
    $('btn_play').textContent = '|| Pause';
    S.playTimer = setInterval(playTick, 100);
  }
  function stopPlay() { clearInterval(S.playTimer); S.playTimer = null; $('btn_play').textContent = '▶ Play'; }
  function playTick() {
    const sl = $('slider_tw'), max = +sl.max;
    const mult = parseInt($('combo_speed').value);
    let v = +sl.value + 10 * mult;
    if (v >= max) { v = max; stopPlay(); }
    sl.value = v; onTwChange();
  }
  function onTwChange() {
    const v = +$('slider_tw').value;
    $('lbl_tw').textContent = v < 0 ? 'All' : (v / 100).toFixed(1);
    refresh();
  }
  function resetTime() { stopPlay(); $('slider_tw').value = -100; onTwChange(); }

  // ---- epoch / embedding swap ----
  // space is 'feat' (featurization) or 'enc' (GNN encoder). Resolved by file
  // identity, not by matching the literal "word2vec" in the name.
  async function selectEmbedding(space) {
    const cur = S.run.file;
    const target = space === 'feat' ? S.run.word2vec_file : S.run.encoder_file;
    if (target && target !== cur) await loadRun(target);
  }
  async function selectEpoch(file) { if (file) await loadRun(file); }

  // ---- search / CSV filter (resolved server-side into buffer-row sets) ----
  async function applySearch() {
    if (!S.run) return;
    const q = $('search_box').value.trim();
    const tok = ++searchToken;
    if (!q) { S.searchRows = null; refresh(); return; }
    try {
      const { rows } = await Data.getSearch(S.run.file, q);
      if (tok !== searchToken) return;   // superseded by newer keystrokes
      S.searchRows = new Set(rows);
    } catch (e) { S.searchRows = null; }
    refresh();
  }
  async function applyCsv(terms) {
    if (!S.run) return;
    if (!terms || !terms.length) { S.csvRows = null; refresh(); return; }
    try {
      const { rows } = await Data.getFilter(S.run.file, terms);
      S.csvRows = new Set(rows);
    } catch (e) { S.csvRows = null; }
    refresh();
  }

  return {
    S, $, loadRun, refresh, selectNode, clearSelection, showLoading,
    togglePlay, stopPlay, onTwChange, resetTime, selectEmbedding, selectEpoch,
    flag, attr, applySearch, applyCsv, computeVisibleInto, TYPE_NAMES,
  };
})();

// Expose the modules for the console / automated tests (harmless in prod).
Object.assign(window, { App, Data, Scene, UI, Dialogs, Campaign, Jobs });

// ---- bootstrap ----
window.addEventListener('DOMContentLoaded', async () => {
  Scene.init(document.getElementById('glcanvas'));
  UI.init();

  // Any modal closes on backdrop click or Escape (not just its Close button).
  // Every modal is a `.modal-bg` backdrop wrapping a `.modal`; clicking the
  // backdrop itself (target === .modal-bg) or hitting Escape clears the mount.
  const mount = document.getElementById('modal_mount');
  mount.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('modal-bg')) mount.innerHTML = '';
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mount.firstChild) mount.innerHTML = '';
  });
  try {
    const data = await Data.getRuns();
    // 1) reopen the last run viewed (if it still exists in artifacts)
    let last = null;
    try { last = localStorage.getItem('pids_last_run'); } catch (e) { }
    if (last && data.runs.some((r) => r.default_file === last)) {
      App.loadRun(last);
    } else {
      // 2) else the newest run that has viz data; else open the browser
      const ready = data.runs.find((r) => r.default_file);
      if (ready) App.loadRun(ready.default_file);
      else UI.openBrowser();
    }
  } catch (e) { UI.openBrowser(); }
});
