import os
import csv
import json
import ast
import argparse

def verify_m2(args, repo_root, html_path, graph_data):
    nodes_path = os.path.join(repo_root, "Ground_Truth", "orthrus", args.dataset, f"{args.dataset}_nodes.csv")
    edges_path = os.path.join(repo_root, "Ground_Truth", "orthrus", args.dataset, f"{args.dataset}_edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print(f"Error: {nodes_path} or edges file does not exist.")
        return

    gt_nodes = set()
    uuid_to_idx = {}
    with open(nodes_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                idx = parts[-1].strip()
                gt_nodes.add(idx)
                uuid_to_idx[parts[0]] = idx
                
    gt_edges_raw = []
    with open(edges_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                src = uuid_to_idx.get(row[0])
                dst = uuid_to_idx.get(row[1])
                if src and dst:
                    gt_edges_raw.append((src, dst, row[2]))
                    
    reversed_ops = {"read", "recvfrom", "event_read", "event_open", "event_recvfrom", "event_execute", "event_recvmsg", "file_read"}
    gt_edges_structural = set()
    for s, d, op in gt_edges_raw:
        if op.lower() in reversed_ops:
            s, d = d, s
        gt_edges_structural.add((s, d, op))

    vis_nodes = {str(n["id"]) for n in graph_data["nodes"] if n["hop"] == 0}
    vis_edges = {(str(l["source"]), str(l["target"]), str(l["label"])) for l in graph_data["links"] if l["hop"] == 0}
    
    print("\n[Nodes]")
    print(f"Ground Truth nodes: {len(gt_nodes)}")
    print(f"Visualized Hop-0 nodes: {len(vis_nodes)}")
    if gt_nodes == vis_nodes:
        print("✅ Node sets match exactly!")
    else:
        print("❌ Node sets mismatch!")
        print("Missing in visualization:", gt_nodes - vis_nodes)
        
    print("\n[Edges]")
    print(f"Ground Truth raw edges (CSV rows): {len(gt_edges_raw)}")
    print(f"Ground Truth unique structural edges (deduplicated): {len(gt_edges_structural)}")
    print(f"Visualized Hop-0 edges: {len(vis_edges)}")
    
    if gt_edges_structural == vis_edges:
        print("✅ Unique edge sets match exactly!")
    else:
        print("❌ Edge sets mismatch!")
        print("Missing in visualization:", gt_edges_structural - vis_edges)


def verify_darpa_tc(args, repo_root, html_path, graph_data):
    from pidsmaker.config.pipeline import get_yml_cfg, get_runtime_required_args
    from pidsmaker.utils.utils import init_database_connection
    from pidsmaker.utils.labelling import get_uuid2nids
    
    mock_args = get_runtime_required_args(return_unknown_args=False, args=["velox", args.dataset])
    cfg = get_yml_cfg(mock_args)
    cur, conn = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)
    
    gt_nids = set()
    for path in cfg.dataset.ground_truth_relative_path:
        full_path = os.path.join(cfg._ground_truth_dir, path)
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r") as f:
            for row in csv.reader(f):
                if row[0] in uuid2nids:
                    gt_nids.add(str(uuid2nids[row[0]]))
                    
    vis_nodes = {str(n["id"]) for n in graph_data["nodes"] if n["hop"] == 0}
    
    print("\n[Nodes]")
    print(f"Ground Truth node IDs (from evaluation CSVs): {len(gt_nids)}")
    print(f"Visualized Hop-0 nodes: {len(vis_nodes)}")
    
    if gt_nids == vis_nodes:
        print("✅ Node sets match exactly! All ground truth evaluation nodes are present in the visualization.")
    else:
        print("❌ Node sets mismatch!")
        print("Missing in visualization:", gt_nids - vis_nodes)
        print("Extra in visualization:", vis_nodes - gt_nids)
        
    print("\n[Edges]")
    print("Note: DARPA TC datasets (like CADETS) do not have ground truth edge CSVs.")
    print("Edges are extracted dynamically from PostgreSQL where BOTH nodes are malicious.")
    print(f"Dynamically extracted Attack Graph Edges: {len([l for l in graph_data['links'] if l['hop'] == 0])}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="M2_user1")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    import sys
    sys.path.append(repo_root)
    
    html_path = os.path.join(repo_root, "artifacts", f"{args.dataset}_campaign_graph.html")
    
    if not os.path.exists(html_path):
        print(f"Error: {html_path} does not exist. Run visualization script first.")
        return

    # Extract embedded JSON graph from HTML
    with open(html_path, "r") as f:
        html_content = f.read()
        
    start_marker = "var fullGraph = "
    end_marker = "var width ="
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        print("Could not find graph data in HTML.")
        return
        
    end_idx = html_content.find(end_marker, start_idx)
    json_str = html_content[start_idx + len(start_marker) : end_idx].strip()
    if json_str.endswith(";"):
        json_str = json_str[:-1]
        
    graph_data = json.loads(json_str)
    print(f"=== Verification for {args.dataset} ===")
    
    if args.dataset.startswith("M2_"):
        verify_m2(args, repo_root, html_path, graph_data)
    else:
        verify_darpa_tc(args, repo_root, html_path, graph_data)

if __name__ == "__main__":
    main()
