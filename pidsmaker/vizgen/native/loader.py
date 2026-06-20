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
            
            # Invalidate old cache if it doesn't have adp or attack_start
            if len(data) < 7 or "adp" not in data[4] or "attack_start_tw" not in data[4] or "run_config" not in data[4] or data[4].get("cache_v") != 4:
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
        if "word2vec" in path:
            adp = 0.0
            disc_score = 0.0
        else:
            if "encoder_epoch_" in path:
                ep_str = os.path.basename(path).split("encoder_epoch_")[1].split("_")[0]
                manifest_path = os.path.join(os.path.dirname(os.path.dirname(path)), "viz_manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                        for ep_data in manifest.get("epochs", []):
                            if str(ep_data.get("epoch")) == str(ep_str):
                                adp = ep_data.get("adp", 0.0)
                                disc_score = ep_data.get("disc_score", 0.0)
                                break
            else:
                # Legacy encoder (Best) - grab from manifest
                manifest_path = os.path.join(os.path.dirname(os.path.dirname(path)), "viz_manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                        if "epochs" in manifest and manifest["epochs"]:
                            sorted_epochs = sorted(manifest["epochs"], key=lambda x: x.get("adp", 0), reverse=True)
                            adp = sorted_epochs[0].get("adp", 0.0)
                            disc_score = sorted_epochs[0].get("disc_score", 0.0)
    except Exception:
        pass
        
    config_text = ""
    try:
        eval_dir = os.path.dirname(os.path.dirname(path))
        cfg_path = os.path.join(eval_dir, "run_config.yml")
        if not os.path.exists(cfg_path):
            # main.py saves to parent of dataset dir (hash dir), so also check there
            cfg_path = os.path.join(os.path.dirname(eval_dir), "run_config.yml")
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, 'r') as f:
                cfg_data = yaml.safe_load(f)
                
                def clean_dict(d):
                    if not isinstance(d, dict):
                        return d
                    cleaned = {}
                    
                    # Identify inactive methods to filter out
                    KNOWN_METHODS = {
                        "alacarte", "doc2vec", "fasttext", "flash", "temporal_rw", "word2vec",
                        "custom_mlp", "gat", "gin", "graph_attention", "magic_gat", "sage", "tgn", "none", "rcaid_gat", "sum_aggregation", "glstm",
                        "few_shot", "predict_edge_contrastive", "predict_edge_type", "predict_node_type", "reconstruct_edge_embeddings", "reconstruct_node_embeddings", "reconstruct_node_features", "reconstruct_masked_features", "predict_masked_struct", "detect_edge_few_shot",
                        "global_batching", "inter_graph_batching", "intra_graph_batching",
                        "edges", "tgn_last_neighbor",
                        "depimpact", "synthetic_attack_naive", "rcaid_pseudo_graph",
                        "kairos_idf_queue", "provnet_lof_queue"
                    }
                    
                    active_method = None
                    if "used_method" in d and isinstance(d["used_method"], str):
                        active_method = d["used_method"]
                    elif "used_methods" in d and isinstance(d["used_methods"], str):
                        active_method = d["used_methods"]
                        
                    for k, v in d.items():
                        if isinstance(k, str) and k.startswith('_'):
                            continue
                        if v is None or v == "" or v == [] or v == {}:
                            continue
                            
                        # Blacklist bloated arrays and irrelevent static DB config
                        if k in ["attack_to_time_window", "ground_truth_relative_path", "train_dates", "test_dates", "val_dates", "unused_dates", "database", "database_all_file", "host", "password", "port", "user", "node_label_features"]:
                            continue
                            
                        # Filter out sibling dictionaries that represent inactive methods
                        if active_method and isinstance(v, dict) and k in KNOWN_METHODS and k != active_method:
                            continue
                            
                        if isinstance(v, dict):
                            v_clean = clean_dict(v)
                            if v_clean:
                                # Don't show dictionaries if their only remaining key is used_method: none
                                if len(v_clean) == 1 and list(v_clean.keys())[0] in ["used_method", "used_methods"] and v_clean[list(v_clean.keys())[0]] == "none":
                                    continue
                                cleaned[k] = v_clean
                        else:
                            cleaned[k] = v
                    return cleaned
                    
                cleaned_cfg = clean_dict(cfg_data)
                config_text = yaml.dump(cleaned_cfg, default_flow_style=False, sort_keys=True)
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
        "attack_start_tw": float('inf'),
        "attack_start_time": "",
        "run_config": config_text,
        "cache_v": 4
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
        tw_idx = p.get("tw_idx", 0)
        tw_label = p.get("tw_label", "")

        if lbl == 0:
            stats["benign"] += 1
            sizes[i] = 3.0
        else:
            stats["malicious"] += 1
            sizes[i] = 5.0
            if tw_idx < stats["attack_start_tw"]:
                stats["attack_start_tw"] = tw_idx
                stats["attack_start_time"] = tw_label
                
            if "process" in ptype or "subject" in ptype:
                stats["mal_proc"] += 1
            elif "file" in ptype:
                stats["mal_file"] += 1
            elif "netflow" in ptype:
                stats["mal_net"] += 1

    # Load campaign mapping if available
    campaign_path = os.path.join(os.path.dirname(path), "campaign_mapping.json")
    if os.path.exists(campaign_path):
        try:
            with open(campaign_path, "r") as f:
                campaign_data = json.load(f)
            n2a = campaign_data.get("node2attacks", {})
            for p in pts:
                nid = str(p.get("node_id"))
                if nid in n2a:
                    p["campaign_ids"] = n2a[nid]
                else:
                    p["campaign_ids"] = []
            stats["num_campaigns"] = campaign_data.get("num_campaigns", 0)
            stats["attack2nodes"] = campaign_data.get("attack2nodes", {})
            print(f"Loaded campaign mapping: {stats['num_campaigns']} campaigns")
        except Exception as e:
            print(f"Failed to load campaign mapping: {e}")

    adj_path = path.replace("_points.json", "_adj.json")
    attack_edges = []
    full_adj = {}
    if os.path.exists(adj_path):
        print(f"Loading adjacency list from {os.path.basename(adj_path)}...")
        with open(adj_path, "r") as f:
            full_adj = json.load(f)

        malicious_nodes = set()
        for p in pts:
            if p.get("label", 0) == 1:
                malicious_nodes.add(str(p["node_id"]))

        edge_set = set()
        for u, neighbors in full_adj.items():
            if u in malicious_nodes:
                for edge in neighbors:
                    v = edge["nb"] if isinstance(edge, dict) else edge
                    if str(v) in malicious_nodes:
                        pair = tuple(sorted([int(u), int(v)]))
                        edge_set.add(pair)
        attack_edges = list(edge_set)
        print(f"Extracted {len(attack_edges)} attack graph edges")
        
        # Do not delete full_adj as we need it
        import gc
        gc.collect()

    res = (pos_hops, colors, sizes, pts, stats, attack_edges, full_adj)
    
    try:
        t1 = time.time()
        with open(cache_path, "wb") as f:
            pickle.dump(res, f)
        print(f"Saved cache to {os.path.basename(cache_path)} in {time.time()-t1:.2f}s")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return res


def resolve_latest_viz_dir(dataset, model=None):
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
        for manifest in manifests:
            eval_dir = os.path.dirname(manifest)
            viz_dir = os.path.join(eval_dir, "viz")
            if os.path.isdir(viz_dir):
                if model:
                    has_w2v = any("word2vec" in f for f in os.listdir(viz_dir))
                    has_enc = any("encoder" in f for f in os.listdir(viz_dir))
                    if model.lower() == "velox" and not has_w2v:
                        continue
                    if model.lower() in ["orthrus", "rcaid"] and not has_enc:
                        continue
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
    for viz_dir in viz_dirs:
        if model:
            has_w2v = any("word2vec" in f for f in os.listdir(viz_dir))
            has_enc = any("encoder" in f for f in os.listdir(viz_dir))
            if model.lower() == "velox" and not has_w2v:
                continue
            if model.lower() in ["orthrus", "rcaid"] and not has_enc:
                continue
        return viz_dir
        
    return None
