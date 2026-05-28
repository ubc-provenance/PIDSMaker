import os
import json
import time
import numpy as np
import pickle
from .constants import get_color

def load_data(path):
    cache_path = path + ".cache.pkl"
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(path):
        print(f"Loading cached data from {os.path.basename(cache_path)}...")
        t0 = time.time()
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            # Invalidate old cache if it doesn't have adp
            if "adp" not in data[4]:
                print("Old cache format detected. Regenerating...")
                raise ValueError("Old cache format")
                
            print(f"Loaded from cache in {time.time()-t0:.2f}s")
            return data
        except Exception as e:
            print(f"Cache load failed: {e}. Falling back to JSON parsing...")

    print(f"Loading {os.path.basename(path)}...")
    t0 = time.time()
    with open(path, "r") as f:
        pts = json.load(f)
    print(f"Loaded {len(pts)} points in {time.time()-t0:.2f}s")

    num_hops = len(pts[0].get("coords_hops", [[0,0,0]])) if "coords_hops" in pts[0] else 1
    pos_hops = [np.zeros((len(pts), 3), dtype=np.float32) for _ in range(num_hops)]
    colors = np.zeros((len(pts), 4), dtype=np.float32)
    sizes = np.zeros(len(pts), dtype=np.float32)

    adp = None
    disc_score = None
    try:
        ep_str = path.split("_epoch_")[1].split("_")[0]
        pr_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "precision_recall_dir")
        stats_path = os.path.join(pr_dir, f"stats_model_epoch_{ep_str}.pkl")
        if os.path.exists(stats_path):
            import torch
            d = torch.load(stats_path, map_location="cpu")
            adp = d.get("adp", 0.0)
            disc_score = d.get("discrimination_score", 0.0)
    except Exception:
        pass

    stats = {
        "total": len(pts),
        "benign": 0,
        "malicious": 0,
        "mal_proc": 0,
        "mal_file": 0,
        "mal_net": 0,
        "adp": adp,
        "disc_score": disc_score,
    }

    for i, p in enumerate(pts):
        if "coords_hops" in p:
            for h in range(num_hops):
                pos_hops[h][i] = p["coords_hops"][h][:3]
        else:
            pos_hops[0][i] = [p.get("x", 0), p.get("y", 0), p.get("z", 0)]
            
        colors[i] = get_color(p)

        lbl = p.get("label", 0)
        ptype = (p.get("type") or "").lower()
        if lbl == 0:
            stats["benign"] += 1
            sizes[i] = 3.0
        else:
            stats["malicious"] += 1
            sizes[i] = 5.0
            if "process" in ptype or "subject" in ptype:
                stats["mal_proc"] += 1
            elif "file" in ptype:
                stats["mal_file"] += 1
            elif "netflow" in ptype:
                stats["mal_net"] += 1

    adj_path = path.replace("_points.json", "_adj.json")
    attack_edges = []
    if os.path.exists(adj_path):
        print(f"Loading adjacency list from {os.path.basename(adj_path)}...")
        with open(adj_path, "r") as f:
            adj = json.load(f)

        malicious_nodes = set()
        for p in pts:
            if p.get("label", 0) == 1:
                malicious_nodes.add(str(p["node_id"]))

        edge_set = set()
        for u, neighbors in adj.items():
            if u in malicious_nodes:
                for v in neighbors:
                    if str(v) in malicious_nodes:
                        pair = tuple(sorted([int(u), int(v)]))
                        edge_set.add(pair)
        attack_edges = list(edge_set)
        print(f"Extracted {len(attack_edges)} attack graph edges")
        
        del adj
        import gc
        gc.collect()

    res = (pos_hops, colors, sizes, pts, stats, attack_edges)
    
    try:
        t1 = time.time()
        with open(cache_path, "wb") as f:
            pickle.dump(res, f)
        print(f"Saved cache to {os.path.basename(cache_path)} in {time.time()-t1:.2f}s")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return res


def resolve_latest_viz_dir(dataset):
    import glob

    pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if os.path.exists("/home/artifacts"):
        artifacts_root = "/home/artifacts"
    else:
        artifacts_root = os.environ.get(
            "PIDS_ARTIFACTS_DIR", os.path.join(pidsmaker_root, "artifacts")
        )

    manifest_patterns = [
        os.path.join(
            artifacts_root, "evaluation/evaluation/*", dataset, "viz_manifest.json"
        ),
        os.path.join(
            artifacts_root, "detection/evaluation/*", dataset, "viz_manifest.json"
        ),
    ]
    manifests = []
    for pattern in manifest_patterns:
        manifests.extend(glob.glob(pattern))
    if manifests:
        manifests.sort(key=os.path.getmtime, reverse=True)
        eval_dir = os.path.dirname(manifests[0])
        viz_dir = os.path.join(eval_dir, "viz")
        if os.path.isdir(viz_dir):
            return viz_dir

    viz_dirs = []
    for base in ("evaluation/evaluation", "detection/evaluation"):
        viz_dirs.extend(
            glob.glob(os.path.join(artifacts_root, base, "*", dataset, "viz"))
        )
    if not viz_dirs:
        viz_dirs = glob.glob(os.path.join(artifacts_root, "viz"))
        if not viz_dirs:
            return None

    viz_dirs.sort(key=os.path.getmtime, reverse=True)
    return viz_dirs[0]
