import json
import csv
import sys

points_file = "/home/artifacts/evaluation/evaluation/d1c76fc7d1448a1c41226e41077119f87271733c6e2a83d2f6c188f1be54ea08/THEIA_E3/viz/embedding_viz_THEIA_E3_word2vec_points.json"

try:
    with open(points_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading points file: {e}")
    sys.exit(1)

test_nodes = []
counts = {"firefox": 0, "systemd": 0, "bash": 0, "malicious": 0}

for p in data:
    path = str(p.get("path", "")).lower()
    cmd = str(p.get("cmd", "")).lower()
    label = p.get("label", 0)
    
    if label == 1 and counts["malicious"] < 50:
        test_nodes.append((p["node_id"], "malicious", path))
        counts["malicious"] += 1
    elif "firefox" in path or "firefox" in cmd:
        if counts["firefox"] < 50:
            test_nodes.append((p["node_id"], "firefox", path))
            counts["firefox"] += 1
    elif "systemd" in path or "systemd" in cmd:
        if counts["systemd"] < 50:
            test_nodes.append((p["node_id"], "systemd", path))
            counts["systemd"] += 1
    elif "bash" in path or "bash" in cmd:
        if counts["bash"] < 50:
            test_nodes.append((p["node_id"], "bash", path))
            counts["bash"] += 1

out_csv = "test_nodes.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "category", "path"])
    for node in test_nodes:
        writer.writerow(node)

print(f"Generated {out_csv} with {len(test_nodes)} nodes:")
for k, v in counts.items():
    print(f"  {k}: {v}")
