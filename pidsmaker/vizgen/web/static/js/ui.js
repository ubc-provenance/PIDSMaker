/* ui.js — DOM construction, panel population, event wiring, run browser */
const UI = (() => {
  const $ = (id) => document.getElementById(id);
  let searchTimer = null;
  let dragStart = null;

  function fmtPct(a, b) { return b ? (100 * a / b).toFixed(1) + '%' : '0%'; }

  function populateEpochs(run) {
    const wrap = $('epoch_wrap'), sel = $('cmb_epoch');
    if (!run.epochs || run.epochs.length === 0) { wrap.classList.add('hidden'); return; }
    wrap.classList.remove('hidden');
    sel.innerHTML = '';
    run.epochs.forEach((e) => {
      const o = document.createElement('option');
      o.value = e.file;
      const metrics = (e.adp || e.disc)
        ? `  ADP ${(e.adp || 0).toFixed(3)} | Disc ${(e.disc || 0).toFixed(3)}` : '';
      o.textContent = `Epoch ${e.epoch}${e.is_best ? ' ★' : ''}${metrics}`;
      // select by file (epoch numbers can repeat: bare "best" file + explicit epoch)
      if (run.file && e.file === run.file) o.selected = true;
      sel.appendChild(o);
    });
    const cur = run.current_epoch != null ? run.current_epoch : (run.best_epoch || '');
    $('lbl_epoch').textContent = 'Loaded: Epoch ' + cur + (run.epochs.find((e) => e.is_best && String(e.epoch) === String(cur)) ? ' (best)' : '');
  }

  function fillStats(run) {
    const s = run.stats;
    $('st_ds').textContent = run.dataset || 'Unknown';
    $('st_perf').textContent = (s.adp || s.disc_score)
      ? `ADP: ${(s.adp || 0).toFixed(3)} | Discrim: ${(s.disc_score || 0).toFixed(3)}` : 'N/A';
    if (s.attack_start_tw !== undefined && s.attack_start_tw >= 0) {
      $('st_attack_row').classList.remove('hidden');
      $('st_attack').textContent = `Window ${s.attack_start_tw}` + (s.attack_start_time ? ` (${s.attack_start_time})` : '');
    } else $('st_attack_row').classList.add('hidden');
    $('st_total').textContent = s.total;
    $('st_benign').textContent = s.benign;
    $('st_mal').textContent = s.malicious;
    $('st_mproc').textContent = s.mal_proc;
    $('st_mnet').textContent = s.mal_net;
    $('st_mfile').textContent = s.mal_file;
    $('lbl_metrics').textContent = $('st_perf').textContent;
  }

  function fillDetection(dc) {
    const set = (id, v) => { $(id).textContent = v; };
    if (!dc) {
      ['dc_gt', 'dc_det', 'dc_fp', 'dc_camp', 'dc_recall', 'dc_fcamp'].forEach((i) => set(i, '—'));
      return;
    }
    set('dc_gt', dc.total_gt ?? '—');
    set('dc_det', dc.total_gt ? `${dc.detected}/${dc.total_gt} (${fmtPct(dc.detected, dc.total_gt)})` : (dc.detected ?? '—'));
    set('dc_fp', dc.current_fp ?? '—');
    set('dc_camp', dc.num_campaigns
      ? `${dc.campaign_coverage_det ?? 0}/${dc.num_campaigns} (${fmtPct(dc.campaign_coverage_det || 0, dc.num_campaigns)})` : '—');
    set('dc_recall', dc.fp_full_recall ?? '—');
    set('dc_fcamp', dc.fp_full_campaign ?? '—');
  }

  // sel = the selected node's server-fetched metadata row (null while it loads;
  // numeric fields always come from the buffer so they show instantly).
  function showInspector(idx, k, sel) {
    const F = App.flag, A = App.attr;
    const lbl = F(idx, 1), det = F(idx, 2);
    const lblTxt = lbl !== 0
      ? `<span style="color:#ef4444">Malicious${det === 2 ? ' (Undetected)' : ' (Detected)'}</span>`
      : '<span style="color:#10b981">Benign</span>';
    const pc = k.purity > 0.8 ? '#10b981' : k.purity > 0.5 ? '#f59e0b' : '#ef4444';
    const gc = k.gap > 3 ? '#10b981' : k.gap > 1 ? '#f59e0b' : '#ef4444';
    const types = Object.entries(k.types).map(([t, c]) => `${t}:${c}`).join(', ');
    const pre = (k.topPre && k.topPre.length)
      ? k.topPre.map(([p, c]) => `  ${c}× ${p || '(none)'}`).join('\n') : '  (loading…)';
    const load = '<span style="color:var(--muted)">loading…</span>';
    // Path first, in the temporal-trajectory blue, so it reads as the primary field.
    $('info_lbl').innerHTML =
      `<span style="color:#60a5fa">Path: ${sel ? (sel.path || '-') : load}</span>\n` +
      `<span style="color:#a0a0ff">ID: ${sel && sel.id != null ? sel.id : (F(idx, 0) || '…')}</span>\n` +
      `Type: ${sel ? (sel.type || 'Unknown') : (App.TYPE_NAMES[F(idx, 3)] || 'Unknown')}\n` +
      `Label: ${lblTxt}\n` +
      `Time Window: ${sel && sel.tw != null ? sel.tw : A(idx, 0)}\n` +
      `Anomaly Score: ${(sel && sel.score != null ? sel.score : A(idx, 3)).toFixed(4)}\n` +
      `Top Edge: ${sel ? (sel.top_edge || '-') : load}\n` +
      `\n── NEIGHBORHOOD (K=${k.K}) ──\n` +
      `Benign: ${k.nb}  Detected: ${k.nd}  Undetected: ${k.nu}\n` +
      `Types: ${types}\n` +
      `Purity: <span style="color:${pc}">${(k.purity * 100).toFixed(0)}%</span>\n` +
      `Score Gap: <span style="color:${gc}">${k.gap.toFixed(2)}</span>\n` +
      `Top paths:\n${pre}`;
  }

  // ---------- modal helpers ----------
  function modalShell(title, bodyHtml, prefix, actionLabel, width) {
    const action = actionLabel ? `<button id="${prefix}_action" class="btn-blue">${actionLabel}</button> ` : '';
    return `<div class="modal-bg"><div class="modal" style="max-width:${width || 920}px">` +
      `<h2>${title}</h2><div class="body">${bodyHtml}</div>` +
      `<div class="foot">${action}<button id="${prefix}_close">Close</button></div></div></div>`;
  }
  function closeModal() { $('modal_mount').innerHTML = ''; }

  // ---------- run browser ----------
  async function openBrowser() {
    $('modal_mount').innerHTML = modalShell('Run Browser',
      '<input type="text" id="rb_filter" placeholder="Filter by dataset / hash / model…" style="margin-bottom:10px">' +
      '<div id="rb_body">Scanning artifacts…</div>', 'rb', null, 1080);
    $('rb_close').onclick = closeModal;
    const data = await Data.getRuns();
    renderRuns(data);
    $('rb_filter').addEventListener('input', () => renderRuns(data, $('rb_filter').value.toLowerCase()));
  }

  function renderRuns(data, filter) {
    if (!data.runs.length) {
      $('rb_body').innerHTML = `No runs found under <code>${data.artifacts_root}</code>.`; return;
    }
    const rows = data.runs.filter((r) => !filter ||
      `${r.dataset} ${r.full_hash} ${r.model}`.toLowerCase().includes(filter));
    const byHash = (h) => data.runs.find((r) => r.full_hash === h);
    const curFile = (App.S.run && App.S.run.file) || '';   // the run open right now
    let h = '<table class="runs"><tr><th>Date</th><th>Dataset</th><th>Model</th><th>Hash</th>' +
      '<th>ADP/Disc</th><th>Status</th><th>Action</th></tr>';
    rows.forEach((r) => {
      const isCur = curFile && r.eval_dir && curFile.startsWith(r.eval_dir + '/');
      const badge = r.status === 'ready' ? '<span class="badge ready">Ready</span>'
        : r.status === 'partial' ? '<span class="badge partial">Partial</span>'
          : '<span class="badge needs">Needs viz</span>';
      const metrics = (r.adp || r.disc_score) ? `${(r.adp || 0).toFixed(3)} / ${(r.disc_score || 0).toFixed(3)}` : '—';
      let act = '';
      if (r.status !== 'needs_viz') act += `<button class="mini act-viz" data-h="${r.full_hash}">Visualize</button> `;
      act += `<button class="mini ${r.status === 'needs_viz' ? 'btn-blue' : ''} act-gen" data-h="${r.full_hash}" ` +
        `title="${r.status === 'ready' ? 'Regenerate' : 'Generate viz data'}">${r.status === 'ready' ? '↻' : 'Generate'}</button>`;
      h += `<tr class="${isCur ? 'current' : ''}"><td>${isCur ? '● ' : ''}${r.date}</td><td>${r.dataset}</td><td>${r.model}</td>` +
        `<td class="mono" title="${r.full_hash}">${r.hash} <span class="copy" data-h="${r.full_hash}" title="copy full hash">⧉</span></td>` +
        `<td>${metrics}</td><td>${badge}${isCur ? ' <span class="badge open">open</span>' : ''}</td><td class="actions">${act}</td></tr>`;
    });
    $('rb_body').innerHTML = h + '</table>';
    $('rb_body').querySelectorAll('.act-viz').forEach((b) => b.onclick = () => {
      closeModal(); App.loadRun(byHash(b.dataset.h).default_file);
    });
    $('rb_body').querySelectorAll('.act-gen').forEach((b) => b.onclick = () => generateModal(byHash(b.dataset.h)));
    $('rb_body').querySelectorAll('.copy').forEach((b) => b.onclick = (e) => {
      e.stopPropagation(); if (navigator.clipboard) navigator.clipboard.writeText(b.dataset.h);
    });
  }

  // ---------- generate-viz options modal ----------
  async function generateModal(run) {
    const st = await Jobs.getStatus();
    if (st.job && st.job.status === 'running') { openConsole(); return; } // already busy → attach
    const present = `Featurization: ${run.present.word2vec ? '✓' : '✗'} · Encoder: ${run.present.encoder ? '✓' : '✗'}`;
    $('modal_mount').innerHTML = modalShell('Generate Viz Data',
      `<div class="gen-summary">${run.dataset} · <span class="mono">${run.hash}</span> · ${run.model}` +
      `<br><span class="muted">${present}</span></div>` +
      '<div class="gen-form">' +
      '<label>Embeddings</label><div>' +
      '<label class="r"><input type="radio" name="emb" value="word2vec">Featurization</label>' +
      '<label class="r"><input type="radio" name="emb" value="encoder">GNN Encoder</label>' +
      '<label class="r"><input type="radio" name="emb" value="both" checked>Both</label></div>' +
      '<label class="chk2"><input type="checkbox" id="g_allep">All epochs (encoder)</label>' +
      '<label>Method</label><div>' +
      '<label class="r"><input type="radio" name="meth" value="umap" checked>UMAP</label>' +
      '<label class="r"><input type="radio" name="meth" value="tsne">t-SNE</label></div>' +
      '<label>Max benign <input type="text" id="g_mb" value="all" class="small"></label>' +
      '<label>Max attack <input type="text" id="g_ma" value="all" class="small"></label>' +
      '</div>', 'gen', 'Start generation');
    $('gen_close').onclick = closeModal;
    $('gen_action').onclick = async () => {
      const mount = $('modal_mount');
      const opts = {
        run_dir: run.eval_dir,
        embeddings: mount.querySelector('input[name=emb]:checked').value,
        all_epochs: $('g_allep').checked,
        method: mount.querySelector('input[name=meth]:checked').value,
        max_benign: $('g_mb').value.trim() || 'all',
        max_attack: $('g_ma').value.trim() || 'all',
      };
      const res = await Jobs.start(opts);
      if (res.error) { alert('Failed to start: ' + res.error); return; }
      openConsole(); // attach to the new (or busy) job
    };
  }

  // ---------- live job console ----------
  function openConsole() {
    $('modal_mount').innerHTML = modalShell('Generating Viz',
      '<div id="cz_phasebar" class="phasebar"></div>' +
      '<div id="cz_meta" class="muted mono"></div>' +
      '<pre id="cz_log" class="console"></pre>', 'cz', null);
    const foot = $('modal_mount').querySelector('.foot');
    foot.insertAdjacentHTML('afterbegin',
      '<button id="cz_cancel" class="btn-red">Cancel</button> ' +
      '<button id="cz_copy">Copy logs</button> ' +
      '<button id="cz_viz" class="btn-blue hidden">Visualize now</button> ');
    $('cz_close').onclick = closeModal; // keeps job + pill running
    $('cz_cancel').onclick = () => Jobs.cancel();
    $('cz_copy').onclick = () => { if (navigator.clipboard) navigator.clipboard.writeText($('cz_log').textContent); };

    let lastI = 0, phases = [], labels = {}, curPhase = 'start', dataset = '', runDir = '';
    let startedAt = Date.now(), timer = null;
    const logEl = $('cz_log');

    function appendLine(i, line) {
      if (i <= lastI) return; lastI = i;
      logEl.textContent += line + '\n'; logEl.scrollTop = logEl.scrollHeight;
    }
    function renderBar() {
      const idx = phases.indexOf(curPhase);
      $('cz_phasebar').innerHTML = phases.map((p, k) =>
        `<span class="step ${k < idx ? 'past' : k === idx ? 'cur' : ''}">${labels[p] || p}</span>`)
        .join('<span class="sep">›</span>');
    }
    function setMeta(extra) {
      const s = ((Date.now() - startedAt) / 1000).toFixed(0);
      $('cz_meta').textContent = `${dataset} · ${s}s${extra ? ' · ' + extra : ''}`;
    }
    function pill() { Jobs.showPill(`Generating viz ${dataset}… ${labels[curPhase] || ''}`); }
    function finish(status) {
      if (timer) { clearInterval(timer); timer = null; }
      setMeta(status.toUpperCase());
      Jobs.setPillState(`Viz ${dataset} ${status}`, status);
      $('cz_cancel').classList.add('hidden');
      if (status === 'done') {
        $('cz_viz').classList.remove('hidden');
        $('cz_viz').onclick = async () => {
          const data = await Data.getRuns();
          const r = data.runs.find((x) => x.eval_dir === runDir);
          closeModal(); Jobs.hidePill();
          if (r && r.default_file) App.loadRun(r.default_file);
          else alert('Generation finished but no points file was found.');
        };
      }
    }

    timer = setInterval(() => setMeta(), 1000);
    Jobs.setReopen(openConsole);
    Jobs.stream((msg) => {
      if (msg.type === 'snapshot') {
        phases = msg.phases || []; labels = msg.phase_labels || {};
        curPhase = msg.phase; dataset = msg.dataset; runDir = msg.run_dir;
        startedAt = Date.now() - (msg.elapsed || 0) * 1000;
        (msg.tail || []).forEach((t) => appendLine(t.i, t.line));
        renderBar(); setMeta();
        if (msg.status && msg.status !== 'running') finish(msg.status); else pill();
      } else if (msg.type === 'log') { appendLine(msg.i, msg.line); }
      else if (msg.type === 'phase') { curPhase = msg.phase; renderBar(); pill(); }
      else if (msg.type === 'status') { finish(msg.status); }
    });
  }

  // ---------- event wiring ----------
  function init() {
    const canvas = $('glcanvas');
    Jobs.reattachOnLoad(openConsole); // reattach pill if an export is running

    // pan sliders
    ['pan_x', 'pan_y', 'pan_z'].forEach((id) => $(id).addEventListener('input', () => {
      Scene.setCenterOffset(+$('pan_x').value / 100, +$('pan_y').value / 100, +$('pan_z').value / 100);
    }));
    // hops
    $('hops').addEventListener('input', () => {
      $('lbl_hops').textContent = `Hops (${$('hops').value}):`;
      Scene.setHop(+$('hops').value); App.refresh();
    });
    $('btn_reset_cam').onclick = () => {
      ['pan_x', 'pan_y', 'pan_z'].forEach((id) => $(id).value = 0);
      Scene.resetCamera();
    };
    $('btn_reset_home').onclick = () => {
      App.clearSelection(); App.resetTime();
      ['pan_x', 'pan_y', 'pan_z'].forEach((id) => $(id).value = 0);
      Scene.resetCamera(); App.refresh();
    };
    $('cmb_epoch').addEventListener('change', (e) => App.selectEpoch(e.target.value));
    $('chk_temporal').addEventListener('change', (e) => {
      Scene.set2D(!e.target.checked); App.refresh();
    });

    // filters
    ['chk_benign', 'chk_det', 'chk_undet'].forEach((id) => $(id).addEventListener('change', App.refresh));
    // overlays
    ['chk_traj', 'chk_attack', 'chk_heat', 'chk_fp_campaign', 'chk_fp_recall']
      .forEach((id) => $(id).addEventListener('change', App.refresh));

    // search (300ms debounce → server-side /api/search)
    $('search_box').addEventListener('input', () => {
      clearTimeout(searchTimer); searchTimer = setTimeout(App.applySearch, 300);
    });
    // csv
    $('btn_load_csv').onclick = () => $('csv_input').click();
    $('csv_input').addEventListener('change', loadCsv);
    $('btn_clear_csv').onclick = () => {
      App.applyCsv(null); $('btn_clear_csv').classList.add('hidden');
      $('lbl_csv_status').classList.add('hidden'); $('lbl_csv_terms').classList.add('hidden');
      App.refresh();
    };

    // playback
    $('btn_play').onclick = App.togglePlay;
    $('btn_reset_tw').onclick = App.resetTime;
    $('slider_tw').addEventListener('input', () => { App.stopPlay(); App.onTwChange(); });

    // overlays buttons
    $('cmb_emb').addEventListener('change', (e) => App.selectEmbedding(e.target.value));
    $('btn_open_browser').onclick = openBrowser;
    $('btn_collapse_leg').onclick = () => {
      const lg = $('legend'); const hidden = lg.classList.toggle('hidden');
      $('btn_collapse_leg').textContent = hidden ? '▼' : '▲';
    };

    // dialogs
    $('btn_causal').onclick = () => Dialogs.causal();
    $('btn_neighbors').onclick = () => Dialogs.neighbors();
    $('btn_plot_dist').onclick = () => Dialogs.scoreDist();
    $('btn_campaign_graph').onclick = () => Campaign.open();

    // picking (5px drag threshold). Use POINTER events, not mouse events:
    // OrbitControls calls preventDefault() on pointerdown, which suppresses
    // the compatibility mousedown/mouseup events — so a mouse-event picker
    // never fires in a real browser. Pointer events are not suppressed.
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button === 0 && e.isPrimary) dragStart = [e.clientX, e.clientY];
    });
    window.addEventListener('pointerup', (e) => {
      if (e.button !== 0 || !dragStart) return;
      const dx = e.clientX - dragStart[0], dy = e.clientY - dragStart[1];
      dragStart = null;
      if (Math.hypot(dx, dy) > 5) return; // was an orbit drag
      if (!App.S.run) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left, py = e.clientY - rect.top;
      if (px < 0 || py < 0 || px > rect.width || py > rect.height) return; // released off-canvas
      // GPU pick first (O(1)); fall back to the CPU scan only on WebGL1 (-2),
      // where we must build the full visibility array for the projection test.
      let idx = Scene.pickGPU(px, py);
      if (idx === -2) idx = Scene.pick(px, py, fullVisible());
      if (idx >= 0) App.selectNode(idx);
      else { App.clearSelection(); App.refresh(); }
    });
  }

  // for picking we want to consider points hidden only by selection too, so
  // reconstruct a visibility that ignores the selection-hide (hideSelected=false).
  function fullVisible() {
    const v = new Float32Array(App.S.n);
    App.computeVisibleInto(v, false);
    return v;
  }

  function loadCsv(e) {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const skip = new Set(['name', 'path', 'id', 'node', 'category']);
      const terms = [];
      reader.result.split(/\r?\n/).forEach((line) => {
        const tok = (line.split(',')[0] || '').trim().toLowerCase();
        if (tok && !skip.has(tok)) terms.push(tok);
      });
      App.applyCsv(terms.length ? terms : null);   // resolved server-side into a node-id set
      $('btn_clear_csv').classList.remove('hidden');
      $('lbl_csv_status').classList.remove('hidden');
      $('lbl_csv_status').textContent = `${terms.length} filter terms loaded`;
      $('lbl_csv_terms').classList.remove('hidden');
      $('lbl_csv_terms').textContent = terms.slice(0, 20).join(', ') + (terms.length > 20 ? ' …' : '');
    };
    reader.readAsText(file);
  }

  return { init, openBrowser, populateEpochs, fillStats, fillDetection, showInspector };
})();
