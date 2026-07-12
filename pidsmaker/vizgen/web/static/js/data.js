/* data.js — API client + binary buffer decoding */
const Data = (() => {
  // half-float (float16) → float32 via a 64K lookup table built once, so decoding
  // the streamed positions is a table read per value, not a bit-twiddle.
  const HALF_LUT = (() => {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
      if (e === 0) t[h] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
      else if (e === 0x1f) t[h] = f ? NaN : (s ? -1 : 1) * Infinity;
      else t[h] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return t;
  })();
  function halfToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = HALF_LUT[u16[i]];
    return out;
  }

  async function getRuns() {
    const r = await fetch('/api/runs');
    return r.json();
  }

  async function getRun(file) {
    const r = await fetch('/api/run?file=' + encodeURIComponent(file));
    if (!r.ok) throw new Error('run load failed: ' + r.status);
    return r.json();
  }

  // Full metadata for a few buffer ROWS, on demand (selection/inspector). Keyed by
  // row index so the client needn't hold the node-id column. Returns {rows:{idx:{
  // id, path, cmd, type, tw, score, top_edge, ...}}}. O(1) per row, instant at 20M.
  async function getNode(file, idxs) {
    const r = await fetch('/api/node?file=' + encodeURIComponent(file) + '&idxs=' + idxs.join(','));
    if (!r.ok) throw new Error('node fetch failed: ' + r.status);
    return r.json();
  }
  // Server-side search: buffer rows matching a query (id or path/cmd substring).
  async function getSearch(file, q) {
    const r = await fetch('/api/search?file=' + encodeURIComponent(file) + '&q=' + encodeURIComponent(q));
    if (!r.ok) throw new Error('search failed: ' + r.status);
    return r.json();
  }
  // Server-side CSV filter: buffer rows matching ANY of many terms.
  async function getFilter(file, terms) {
    const r = await fetch('/api/filter', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file, terms }),
    });
    if (!r.ok) throw new Error('filter failed: ' + r.status);
    return r.json();
  }

  async function fetchRange(url, start, len) {
    const r = await fetch(url, { headers: { Range: `bytes=${start}-${start + len - 1}` } });
    if (r.status !== 206 && r.status !== 200) throw new Error('range fetch ' + r.status);
    return r.arrayBuffer();
  }

  /* Decode nodes [start, start+count) of the binary buffer via HTTP range
     requests per column — so a huge run loads in bounded chunks, independent of
     the full node count. Returns the same shape as a full buffer, with n=count. */
  // Tiered buffer fetch (v4 columns: positions f16 | attrs u16×4 | ids u32 | meta u8).
  // TIER 1 (core) = positions + meta: everything needed to render the cloud (colour
  // and size derive from the meta byte). Fetched first so the cloud paints ASAP.
  async function getBufferCore(run, start, count) {
    const url = '/api/buffer?file=' + encodeURIComponent(run.file);
    const n = run.n, H = run.hops, off = run.byte_offsets;
    const reqs = [];
    for (let h = 0; h < H; h++) reqs.push(fetchRange(url, off.positions + (h * n + start) * 3 * 2, count * 3 * 2));
    reqs.push(fetchRange(url, off.meta + start, count));
    const b = await Promise.all(reqs);
    const pos = [];
    for (let h = 0; h < H; h++) pos.push(halfToF32(new Uint16Array(b[h])));
    return { n: count, H, pos, meta: new Uint8Array(b[H]) };
  }
  // TIER 2 (aux) = attrs (packed) + ids: time animation, playback, overlays, attack
  // graph, and the id→instances map. Loaded in the background after the cloud is up.
  // attrs are unpacked into the same f32[n*5] shape (tw_idx, tw_start, tw_end,
  // score, size) the rest of the app expects; size is left 0 (derived from label).
  async function getBufferAux(run, start, count) {
    const url = '/api/buffer?file=' + encodeURIComponent(run.file);
    const off = run.byte_offsets;
    const [ba, bi] = await Promise.all([
      fetchRange(url, off.attrs + start * 4 * 2, count * 4 * 2),
      fetchRange(url, off.ids + start * 4, count * 4),
    ]);
    const packed = new Uint16Array(ba), attrs = new Float32Array(count * 5);
    for (let i = 0; i < count; i++) {
      const p = i * 4, o = i * 5, twEnd = packed[p + 2];
      attrs[o] = packed[p];                                   // tw_idx
      attrs[o + 1] = packed[p + 1];                           // tw_start
      attrs[o + 2] = twEnd === 65535 ? 1e30 : twEnd;          // tw_end (65535 = never)
      attrs[o + 3] = HALF_LUT[packed[p + 3]];                 // score (float16)
      // attrs[o+4] (size) stays 0 — the app uses the label-derived size instead
    }
    return { n: count, attrs, ids: new Uint32Array(bi) };
  }

  async function getNeighbors(file, node) {
    const r = await fetch('/api/neighbors?file=' + encodeURIComponent(file) + '&node=' + encodeURIComponent(node));
    if (!r.ok) throw new Error('neighbors failed: ' + r.status);
    return r.json();
  }
  async function getCausal(file, node) {
    const r = await fetch('/api/causal?file=' + encodeURIComponent(file) + '&node=' + encodeURIComponent(node));
    if (!r.ok) throw new Error('causal failed: ' + r.status);
    return r.json();
  }
  async function getAttackPairs(file) {
    const r = await fetch('/api/attack_pairs?file=' + encodeURIComponent(file));
    if (!r.ok) throw new Error('attack_pairs failed: ' + r.status);
    return r.json();
  }

  return { getRuns, getRun, getNode, getSearch, getFilter, getBufferCore, getBufferAux, getNeighbors, getCausal, getAttackPairs };
})();
