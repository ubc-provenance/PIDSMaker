import torch
import sys

try:
    scores_path = "/home/artifacts/evaluation/evaluation/4574918f46a838062313817bb24ccfa96cac7190283e1dc69e333b0848b8af3c/CADETS_E3/precision_recall_dir/scores_model_epoch_11.pkl"
    data = torch.load(scores_path, map_location="cpu")
    y_preds = data.get("y_preds", [])
    scores = data.get("pred_scores", [])
    
    involved = set()
    
    if "edges" in data:
        edges = data["edges"]
        for i in range(len(y_preds)):
            u, v = int(edges[i][0]), int(edges[i][1])
            if y_preds[i]:
                involved.add(u)
                involved.add(v)
        print(f"Edge mode: {len(involved)} nodes involved.")
    elif "nodes" in data:
        nodes = data["nodes"]
        for i in range(len(y_preds)):
            u = int(nodes[i])
            if y_preds[i]:
                involved.add(u)
        print(f"Node mode: {len(involved)} nodes involved.")
    
    print("Test passed.")
except Exception as e:
    print(f"Failed: {e}")
