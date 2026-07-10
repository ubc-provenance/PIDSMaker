/* jobs.js — viz-generation job transport: start, SSE stream, cancel, status,
   and a persistent status pill that survives page refresh. Rendering of the
   console/phase-bar lives in ui.js; this module is pure transport + pill. */
const Jobs = (() => {
  let pill = null;
  let reopen = null; // callback to reopen the console (set by ui.js)

  async function start(opts) {
    const r = await fetch('/api/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    });
    const data = await r.json().catch(() => ({}));
    if (r.status === 409) return { busy: true, job_id: data.job_id };
    if (!r.ok) return { error: data.error || ('HTTP ' + r.status) };
    return { job_id: data.job_id };
  }

  async function getStatus() {
    try { const r = await fetch('/api/export/status'); return r.json(); }
    catch (e) { return { job: null }; }
  }

  async function cancel() {
    try { await fetch('/api/export/cancel', { method: 'POST' }); } catch (e) {}
  }

  /* Open the SSE stream. onMsg receives parsed objects:
     {type:'snapshot',...} | {type:'log',i,line} | {type:'phase',phase} | {type:'status',status,returncode}
     Returns the EventSource so the caller can .close(). */
  function stream(onMsg) {
    const es = new EventSource('/api/export/stream');
    es.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch (_) { return; }
      onMsg(msg);
      if (msg.type === 'status' && msg.status !== 'running') es.close();
    };
    es.onerror = () => { /* stream ends on completion; ignore */ };
    return es;
  }

  // ---- status pill (bottom-right, left of the model overlay) ----
  function showPill(text) {
    if (!pill) {
      pill = document.createElement('div');
      pill.id = 'job_pill';
      pill.onclick = () => { if (reopen) reopen(); };
      document.body.appendChild(pill);
    }
    pill.innerHTML = '<span class="spin"></span><span class="pill-txt"></span>';
    pill.querySelector('.pill-txt').textContent = text;
    pill.classList.remove('done', 'failed');
  }
  function setPillState(text, state) {
    if (!pill) showPill(text);
    pill.querySelector('.pill-txt').textContent = text;
    const sp = pill.querySelector('.spin');
    pill.classList.toggle('done', state === 'done');
    pill.classList.toggle('failed', state === 'failed' || state === 'canceled');
    if (sp && state !== 'running') sp.style.display = 'none';
  }
  function hidePill() { if (pill) { pill.remove(); pill = null; } }
  function setReopen(fn) { reopen = fn; }

  // On page load: if a job is running, show the pill so the user can reattach.
  async function reattachOnLoad(reopenFn) {
    setReopen(reopenFn);
    const { job } = await getStatus();
    if (job && job.status === 'running') {
      showPill(`Generating viz ${job.dataset}… ${job.phase_labels[job.phase] || ''}`);
    }
  }

  return { start, getStatus, cancel, stream, showPill, setPillState, hidePill, setReopen, reattachOnLoad };
})();
