import os
import sys
import argparse
import json
import csv
import ast
import time
import networkx as nx
import glob

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)


def get_node_color(ntype, is_malicious):
    if not is_malicious:
        return '#808080'  # Grey
    ntype = ntype.lower()
    if 'process' in ntype or 'subject' in ntype:
        return '#ff4d4d'  # Red
    elif 'net' in ntype or 'socket' in ntype:
        return '#4da6ff'  # Blue
    elif 'file' in ntype:
        return '#4dff4d'  # Green
    else:
        return '#ffcc00'  # Yellow


HOP_COLORS = {0: '#ff4d4d', 1: '#ffb3b3', 2: '#cccccc', 3: '#666666'}


def ensure_db_indexes(cur):
    """Create indexes on event_table if they don't exist. One-time cost (~2-3 min), makes all queries instant after."""
    idx_defs = [
        ("idx_event_src", "CREATE INDEX IF NOT EXISTS idx_event_src ON event_table (src_index_id);"),
        ("idx_event_dst", "CREATE INDEX IF NOT EXISTS idx_event_dst ON event_table (dst_index_id);"),
        ("idx_event_ts",  "CREATE INDEX IF NOT EXISTS idx_event_ts  ON event_table (timestamp_rec);"),
    ]
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'event_table';")
    existing = {r[0] for r in cur.fetchall()}

    needed = [(name, sql) for name, sql in idx_defs if name not in existing]
    if not needed:
        print("  ✓ DB indexes already exist.")
        return

    print(f"  ⏳ Creating {len(needed)} index(es) on event_table (one-time cost, ~2-3 min)...")
    for name, sql in needed:
        t0 = time.time()
        print(f"    Creating {name}...", end="", flush=True)
        cur.execute(sql)
        cur.connection.commit()
        print(f" done ({time.time() - t0:.1f}s)")
    print("  ✓ Indexes created. Future queries will be instant.")


def load_m2_ground_truth(args, nodes_dict, links_list, hop0_nodes):
    """Load ground truth directly from M2 CSV files — no database needed."""
    nodes_path = os.path.join(repo_root, "Ground_Truth", "orthrus", args.dataset, f"{args.dataset}_nodes.csv")
    edges_path = os.path.join(repo_root, "Ground_Truth", "orthrus", args.dataset, f"{args.dataset}_edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"Error: Could not find M2 ground truth files at {nodes_path} or {edges_path}")
        sys.exit(1)

    uuid_to_index = {}

    with open(nodes_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                uid = parts[0]
                start = line.find('{')
                end = line.find('}')
                if start != -1 and end != -1:
                    dict_str = line[start:end + 1]
                    try:
                        label_dict = ast.literal_eval(dict_str)
                        ntype = list(label_dict.keys())[0]
                        name = label_dict[ntype]
                    except Exception:
                        ntype, name = "unknown", "unknown"
                else:
                    ntype, name = "unknown", "unknown"

                idx_id = parts[-1].strip()
                uuid_to_index[uid] = idx_id
                hop0_nodes.add(idx_id)

                if len(name) > 40:
                    name = "..." + name[-37:]

                nodes_dict[idx_id] = {
                    "id": idx_id,
                    "label": f"{ntype}: {name}",
                    "hop": 0,
                    "color": get_node_color(ntype, True),
                    "ntype": ntype,
                }

    with open(edges_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                src = uuid_to_index.get(row[0])
                dst = uuid_to_index.get(row[1])
                if src and dst:
                    links_list.append({"source": src, "target": dst, "label": row[2], "hop": 0, "color": HOP_COLORS[0]})

    print(f"  ✓ Loaded {len(hop0_nodes)} nodes, {len(links_list)} edges from CSV.")


def load_darpa_ground_truth(args, nodes_dict, links_list, hop0_nodes, max_hops):
    """Load ground truth from PostgreSQL for DARPA TC datasets (CADETS, THEIA, etc.)."""
    from pidsmaker.config.pipeline import get_yml_cfg, get_runtime_required_args
    from pidsmaker.utils.utils import init_database_connection, datetime_to_ns_time_US
    from pidsmaker.utils.labelling import get_uuid2nids
    import torch

    mock_args = get_runtime_required_args(return_unknown_args=False, args=["velox", args.dataset])
    cfg = get_yml_cfg(mock_args)
    cur, conn = init_database_connection(cfg)

    print("  Checking DB indexes...")
    ensure_db_indexes(cur)

    # --- Load node UUID -> index_id mapping ---
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    def label_node(nid, hop):
        # We will initialize as unknown and update via SQL at the end
        is_mal = (hop == 0)
        return {"id": str(nid), "label": f"unknown: {nid}", "hop": hop, "color": get_node_color("unknown", is_mal), "ntype": "unknown"}

    # --- Hop 0: get malicious edges within attack time windows ---
    global_start_ns, global_end_ns = None, None

    for path, attack_tw in zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window):
        start_ns = datetime_to_ns_time_US(attack_tw[1])
        end_ns = datetime_to_ns_time_US(attack_tw[2])

        if global_start_ns is None or int(start_ns) < int(global_start_ns):
            global_start_ns = start_ns
        if global_end_ns is None or int(end_ns) > int(global_end_ns):
            global_end_ns = end_ns

        gt_nids = set()
        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
            for row in csv.reader(f):
                if row[0] in uuid2nids:
                    gt_nids.add(str(uuid2nids[row[0]]))

        if not gt_nids:
            continue
        nids_str = ",".join(f"'{n}'" for n in gt_nids)

        t0 = time.time()
        print(f"  Hop-0 query for {path}...", end="", flush=True)
        cur.execute(
            f"SELECT src_index_id, dst_index_id, operation FROM event_table "
            f"WHERE timestamp_rec BETWEEN '{start_ns}' AND '{end_ns}' "
            f"AND (src_index_id IN ({nids_str}) OR dst_index_id IN ({nids_str}));"
        )
        rows = cur.fetchall()

        matched = 0
        for r in rows:
            s, d, op = str(r[0]), str(r[1]), str(r[2])
            if s in gt_nids and d in gt_nids:  # MUST be AND so we don't fetch thousands of benign interactions!
                hop0_nodes.add(s)
                hop0_nodes.add(d)
                links_list.append({"source": s, "target": d, "label": op, "hop": 0, "color": HOP_COLORS[0]})
                matched += 1
                
        print(f" {matched} attack edges ({time.time() - t0:.1f}s)")

    # Label hop-0 nodes
    for nid in hop0_nodes:
        nodes_dict[str(nid)] = label_node(nid, 0)

    print(f"  ✓ Hop-0: {len(hop0_nodes)} nodes, {len(links_list)} edges.")

    # --- Hop 1, 2, 3: expand neighborhood ---
    if max_hops >= 1:
        prev_frontier = hop0_nodes
        all_seen = set(hop0_nodes)

        for hop in range(1, max_hops + 1):
            if not prev_frontier:
                break

            nodes_str = ",".join(f"'{n}'" for n in prev_frontier)
            time_clause = f"timestamp_rec BETWEEN '{global_start_ns}' AND '{global_end_ns}' AND " if global_start_ns else ""

            t0 = time.time()
            print(f"  Hop-{hop} query ({len(prev_frontier)} frontier nodes)...", end="", flush=True)
            limit_clause = f"LIMIT {args.hop_limit}" if args.hop_limit > 0 else ""
            cur.execute(
                f"SELECT src_index_id, dst_index_id, operation FROM event_table "
                f"WHERE {time_clause}(src_index_id IN ({nodes_str}) OR dst_index_id IN ({nodes_str})) "
                f"{limit_clause};"
            )
            rows = cur.fetchall()
            print(f" {len(rows)} edges ({time.time() - t0:.1f}s)")

            new_frontier = set()
            for r in rows:
                s, d, op = str(r[0]), str(r[1]), str(r[2])
                for n in (s, d):
                    if n not in all_seen:
                        new_frontier.add(n)
                        nodes_dict[n] = label_node(n, hop)
                # Only add edge if it connects previous frontier to new nodes
                if (s in prev_frontier and d not in all_seen) or (d in prev_frontier and s not in all_seen):
                    links_list.append({"source": s, "target": d, "label": op, "hop": hop, "color": HOP_COLORS.get(hop, '#444444')})

            all_seen |= new_frontier
            prev_frontier = new_frontier
            print(f"  ✓ Hop-{hop}: +{len(new_frontier)} new nodes.")

    # --- Update nodes with real labels from DB ---
    all_seen_nodes = set(nodes_dict.keys())
    if all_seen_nodes:
        print("  Fetching true node labels from DB...", end="", flush=True)
        t0 = time.time()
        nids_str = ",".join(f"'{n}'" for n in all_seen_nodes)
        db_labels = {}
        
        try:
            cur.execute(f"SELECT index_id, path, cmd FROM subject_node_table WHERE index_id IN ({nids_str})")
            for idx, path, cmd in cur.fetchall():
                db_labels[str(idx)] = ("process", f"{path} {cmd}".strip())
                
            cur.execute(f"SELECT index_id, path FROM file_node_table WHERE index_id IN ({nids_str})")
            for idx, path in cur.fetchall():
                db_labels[str(idx)] = ("file", str(path))
                
            cur.execute(f"SELECT index_id, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table WHERE index_id IN ({nids_str})")
            for idx, sa, sp, da, dp in cur.fetchall():
                db_labels[str(idx)] = ("netflow", f"{sa}:{sp} -> {da}:{dp}")
        except Exception as e:
            print(f"\n    (Warning: Error fetching some DB labels: {e})", end="")

        for nid, node in nodes_dict.items():
            if nid in db_labels:
                ntype, name = db_labels[nid]
                if len(name) > 60: name = "..." + name[-57:]
                node["ntype"] = ntype
                node["label"] = f"{ntype}: {name}"
                node["color"] = get_node_color(ntype, node["hop"] == 0)
                
        print(f" done ({time.time() - t0:.1f}s)")


def build_dag(links_list):
    """Reverse read-like edges for Subject->Object orientation and deduplicate to prevent UI lag."""
    reversed_ops = {"read", "recvfrom", "event_read", "event_open", "event_recvfrom",
                    "event_execute", "event_recvmsg", "file_read"}
    
    clean = []
    seen = set()
    
    for link in links_list:
        src, dst = str(link["source"]), str(link["target"])
        label = str(link["label"])
        
        if label.lower() in reversed_ops:
            src, dst = dst, src
            
        key = (src, dst, label)
        if key not in seen:
            seen.add(key)
            link["source"] = src
            link["target"] = dst
            clean.append(link)

    return clean


def generate_html(dataset, graph_data):
    return '''<!DOCTYPE html>
<meta charset="utf-8">
<style>
body { font-family: sans-serif; margin: 0; overflow: hidden; background-color: #1a1a1a; color: #fff;}
.links line { stroke-opacity: 0.8; stroke-width: 2px; }
.nodes circle { stroke: #333; stroke-width: 1.5px; }
.node-label { font-size: 11px; fill: #eee; pointer-events: none; text-shadow: 1px 1px 2px black; }
.edge-label { font-size: 9px; fill: #aaa; pointer-events: none; }
#controls { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }
select { background: #333; color: white; padding: 5px; border: 1px solid #555; border-radius: 3px; font-size: 14px; }
.legend-item { display: inline-block; margin-right: 10px; }
</style>
<script src="https://d3js.org/d3.v4.min.js"></script>
<body>
<div id="controls">
  <h3>''' + dataset + ''' Campaign Graph</h3>
  <label for="hop-select">Show neighborhood up to:</label>
  <select id="hop-select" onchange="updateGraph()">
    <option value="0" selected>0-Hop (Attack Graph Only)</option>
    <option value="1">1-Hop (Immediate Context)</option>
    <option value="2">2-Hop (Extended Context)</option>
    <option value="3">3-Hop (Broad Context)</option>
  </select>
  <p><b>Nodes:</b>
    <span class="legend-item"><span style="color:#ff4d4d;">⬤</span> Process/Subject</span>
    <span class="legend-item"><span style="color:#4dff4d;">⬤</span> File</span>
    <span class="legend-item"><span style="color:#4da6ff;">⬤</span> Netflow</span>
    <span class="legend-item"><span style="color:#808080;">⬤</span> Benign Context</span>
  </p>
  <p><b>Edges:</b>
    <span class="legend-item"><span style="color:#ff4d4d;">—</span> 0-Hop (Attack)</span>
    <span class="legend-item"><span style="color:#ffb3b3;">—</span> 1-Hop</span>
    <span class="legend-item"><span style="color:#cccccc;">—</span> 2-Hop</span>
    <span class="legend-item"><span style="color:#666666;">—</span> 3-Hop</span>
  </p>
  <p>Scroll to zoom, drag to pan</p>
</div>
<script>
var fullGraph = ''' + json.dumps(graph_data) + ''';
var width = window.innerWidth, height = window.innerHeight;

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(d3.zoom().on("zoom", function () {
       svgGroup.attr("transform", d3.event.transform)
    }));

var svgGroup = svg.append("g");

var markerColors = ["#ff4d4d", "#ffb3b3", "#cccccc", "#666666"];
svgGroup.append("defs").selectAll("marker")
    .data(markerColors)
  .enter().append("marker")
    .attr("id", function(d) { return "arrow-" + d.replace('#', ''); })
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 15)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", function(d) { return d; });

var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(80))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(20));

var linkGroup = svgGroup.append("g").attr("class", "links");
var edgeLabelGroup = svgGroup.append("g").attr("class", "edge-labels");
var nodeGroup = svgGroup.append("g").attr("class", "nodes");

var link, edgeLabel, node;

function updateGraph() {
    var maxHop = parseInt(document.getElementById("hop-select").value);

    var nodes = fullGraph.nodes.filter(n => n.hop <= maxHop);
    var nodeIds = new Set(nodes.map(n => n.id));
    var links = fullGraph.links.filter(l => l.hop <= maxHop && nodeIds.has(l.source.id || l.source) && nodeIds.has(l.target.id || l.target));

    var getSourceId = d => (typeof d.source === 'object' ? d.source.id : d.source);
    var getTargetId = d => (typeof d.target === 'object' ? d.target.id : d.target);
    var linkKey = d => getSourceId(d) + "-" + getTargetId(d);

    link = linkGroup.selectAll("line").data(links, linkKey);
    link.exit().remove();
    var linkEnter = link.enter().append("line")
        .attr("stroke", d => d.color)
        .attr("marker-end", d => "url(#arrow-" + d.color.replace('#', '') + ")");
    link = linkEnter.merge(link);

    edgeLabel = edgeLabelGroup.selectAll("text").data(links, linkKey);
    edgeLabel.exit().remove();
    var edgeLabelEnter = edgeLabel.enter().append("text")
        .attr("class", "edge-label")
        .attr("dy", -3)
        .text(d => d.label);
    edgeLabel = edgeLabelEnter.merge(edgeLabel);

    node = nodeGroup.selectAll("g").data(nodes, d => d.id);
    node.exit().remove();

    var nodeEnter = node.enter().append("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    nodeEnter.append("circle")
        .attr("r", 6)
        .attr("fill", d => d.color);

    nodeEnter.append("text")
        .attr("class", "node-label")
        .attr("x", 8)
        .attr("y", 3)
        .text(d => d.label);

    nodeEnter.append("title")
        .text(d => d.label);

    node = nodeEnter.merge(node);

    simulation.nodes(nodes).on("tick", ticked);
    simulation.force("link").links(links);
    simulation.alpha(1).restart();
}

function ticked() {
    link.attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    edgeLabel.attr("x", d => (d.source.x + d.target.x) / 2)
             .attr("y", d => (d.source.y + d.target.y) / 2);

    node.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
}

function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}
function dragged(d) {
    d.fx = d3.event.x; d.fy = d3.event.y;
}
function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}

updateGraph();
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Visualize Campaign Connectivity from Ground Truth")
    parser.add_argument("--dataset", type=str, default="M2_user2", help="Dataset name (e.g. M2_user2, CADETS_E3)")
    parser.add_argument("--hops", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Max neighborhood hops to fetch from DB (default: 0 = attack graph only, no extra DB queries)")
    parser.add_argument("--hop_limit", type=int, default=2000,
                        help="Maximum number of edges to fetch per hop to prevent browser lag (default: 2000, set to 0 for no limit)")
    parser.add_argument("--out_html", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    out_html = args.out_html or f'{repo_root}/artifacts/{args.dataset}_campaign_graph.html'
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    nodes_dict = {}
    links_list = []
    hop0_nodes = set()

    t_total = time.time()

    if args.dataset.startswith("M2_"):
        print(f"\n[M2 Dataset: {args.dataset}]")
        load_m2_ground_truth(args, nodes_dict, links_list, hop0_nodes)
    else:
        print(f"\n[DARPA TC Dataset: {args.dataset}]")
        try:
            load_darpa_ground_truth(args, nodes_dict, links_list, hop0_nodes, max_hops=args.hops)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nFailed to load dataset from DB: {e}")
            sys.exit(1)

    # DAG enforcement
    print("  Building DAG...")
    clean_links = build_dag(links_list)
    graph_data = {"nodes": list(nodes_dict.values()), "links": clean_links}

    # Generate HTML
    html = generate_html(args.dataset, graph_data)
    with open(out_html, 'w') as f:
        f.write(html)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 50}")
    print(f"  ✅ Done in {elapsed:.1f}s — {len(nodes_dict)} nodes, {len(clean_links)} edges")
    print(f"  👉 file://{out_html}")
    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    main()
