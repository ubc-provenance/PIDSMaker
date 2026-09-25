/* campaign.js — Campaign Attack Graph (D3 force-directed provenance view).
   The graph is built on the server (/api/campaign) from the run's own adjacency
   (edges with relation type and direction) and ground-truth node labels/paths:
   hop 0 = malicious<->malicious edges, hops 1/2/3 add context via edge sampling
   (300/200/100), DAG-enforced. This module just renders the returned
   {nodes, links}. */
const Campaign = (() => {
  const $ = (id) => document.getElementById(id);
  const HOP_COLOR = ['#ff4d4d', '#ffb3b3', '#cccccc', '#666666'];
  let sim, link, edgeLabel, node, svgGroup, full;

  async function open() {
    if (!App.S.run) return;
    buildModal();
    $('cg_counts').textContent = 'Building campaign graph…';
    let data;
    try {
      const r = await fetch('/api/campaign?file=' + encodeURIComponent(App.S.run.file));
      if (r.status === 404) { $('cg_counts').textContent = 'No adjacency data for this run. Generate viz data first.'; return; }
      if (!r.ok) { $('cg_counts').textContent = 'Failed to build campaign graph (' + r.status + ').'; return; }
      data = await r.json();
    } catch (e) { $('cg_counts').textContent = 'Error: ' + e.message; return; }
    full = data;
    setupSvg();
    updateGraph(+$('cg_hop').value);
    $('cg_hop').onchange = () => updateGraph(+$('cg_hop').value);
  }

  function buildModal() {
    document.getElementById('modal_mount').innerHTML =
      '<div class="modal-bg"><div class="modal" style="max-width:1280px;width:94%;height:88vh">' +
      '<h2>Campaign Attack Graph</h2>' +
      '<div class="body" style="padding:8px 12px;flex:1;display:flex;flex-direction:column">' +
      '<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;flex-wrap:wrap">' +
      '<label>Show neighborhood up to: <select id="cg_hop">' +
      '<option value="0" selected>0-Hop (Attack Graph Only)</option>' +
      '<option value="1">1-Hop (Immediate Context)</option>' +
      '<option value="2">2-Hop (Extended Context)</option>' +
      '<option value="3">3-Hop (Broad Context)</option></select></label>' +
      '<span id="cg_counts" class="muted mono"></span>' +
      '<span class="cg-legend" style="margin-left:auto">' +
      '<span style="color:#ffcc00">●</span> Subject &nbsp;' +
      '<span style="color:#ff4d4d">●</span> Process &nbsp;' +
      '<span style="color:#4dff4d">●</span> File &nbsp;' +
      '<span style="color:#4da6ff">●</span> Netflow &nbsp;' +
      '<span style="color:#808080">●</span> Context</span>' +
      '<span class="muted" style="font-size:11px;width:100%">Scroll to zoom, drag nodes to pan</span>' +
      '</div>' +
      '<svg id="cg_svg" style="flex:1;width:100%;background:#1a1a1a;border-radius:6px"></svg>' +
      '</div><div class="foot"><button id="cg_close">Close</button></div></div></div>';
    $('cg_close').onclick = () => { document.getElementById('modal_mount').innerHTML = ''; };
  }

  function setupSvg() {
    const el = $('cg_svg'), r = el.getBoundingClientRect();
    const W = r.width || window.innerWidth * 0.8, H = r.height || window.innerHeight * 0.6;
    const svg = d3.select(el);
    svg.selectAll('*').remove();
    svgGroup = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.05, 8]).on('zoom', () => svgGroup.attr('transform', d3.event.transform)));

    svgGroup.append('defs').selectAll('marker').data(HOP_COLOR).enter().append('marker')
      .attr('id', (d) => 'cg-arrow-' + d.replace('#', '')).attr('viewBox', '0 -5 10 10')
      .attr('refX', 15).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto').append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', (d) => d);

    sim = d3.forceSimulation()
      .force('link', d3.forceLink().id((d) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide().radius(20));

    svgGroup.append('g').attr('class', 'links');
    svgGroup.append('g').attr('class', 'edge-labels');
    svgGroup.append('g').attr('class', 'nodes');
  }

  function updateGraph(maxHop) {
    const nodes = full.nodes.filter((n) => n.hop <= maxHop);
    const ids = new Set(nodes.map((n) => n.id));
    const links = full.links.filter((l) => l.hop <= maxHop
      && ids.has(l.source.id || l.source) && ids.has(l.target.id || l.target));
    $('cg_counts').textContent = `${nodes.length} nodes, ${links.length} edges`;
    const lkey = (d) => (d.source.id || d.source) + '-' + (d.target.id || d.target) + '-' + d.label;

    link = svgGroup.select('.links').selectAll('line').data(links, lkey);
    link.exit().remove();
    link = link.enter().append('line')
      .attr('stroke', (d) => d.color).attr('stroke-width', 2).attr('stroke-opacity', 0.8)
      .attr('marker-end', (d) => 'url(#cg-arrow-' + d.color.replace('#', '') + ')').merge(link);

    edgeLabel = svgGroup.select('.edge-labels').selectAll('text').data(links, lkey);
    edgeLabel.exit().remove();
    edgeLabel = edgeLabel.enter().append('text').attr('dy', -3).attr('font-size', 9)
      .attr('fill', '#aaa').attr('text-anchor', 'middle').text((d) => d.label).merge(edgeLabel);

    node = svgGroup.select('.nodes').selectAll('g').data(nodes, (d) => d.id);
    node.exit().remove();
    const ne = node.enter().append('g')
      .call(d3.drag().on('start', dstart).on('drag', dged).on('end', dend));
    ne.append('circle').attr('r', 6).attr('stroke', '#333').attr('stroke-width', 1.5).attr('fill', (d) => d.color);
    ne.append('text').attr('x', 8).attr('y', 3).attr('font-size', 11).attr('fill', '#eee').attr('class', 'cg-nl').text((d) => d.label);
    ne.append('title').text((d) => d.label);
    node = ne.merge(node);

    sim.nodes(nodes).on('tick', ticked);
    sim.force('link').links(links);
    sim.alpha(1).restart();
  }

  function ticked() {
    link.attr('x1', (d) => d.source.x).attr('y1', (d) => d.source.y)
      .attr('x2', (d) => d.target.x).attr('y2', (d) => d.target.y);
    edgeLabel.attr('x', (d) => (d.source.x + d.target.x) / 2).attr('y', (d) => (d.source.y + d.target.y) / 2);
    node.attr('transform', (d) => 'translate(' + d.x + ',' + d.y + ')');
  }

  function dstart(d) { if (!d3.event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
  function dged(d) { d.fx = d3.event.x; d.fy = d3.event.y; }
  function dend(d) { if (!d3.event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }

  return { open };
})();
