"""Standalone web visualizer server for PIDSMaker embeddings.

Flask app that serves the 3D UMAP embedding viewer over localhost. It is
*independent of the pipeline*: it only reads pre-computed artifacts from the
artifacts folder (``embedding_viz_*_points.json`` / ``*_adj.json`` and the
sibling ``viz_manifest.json``). The user picks a run in the browser.

Endpoints
---------
GET  /                       -> single-page app (run browser + 3D viewer)
GET  /api/runs               -> list of discovered runs (artifacts scan)
GET  /api/run?file=<points>  -> run descriptor: n, hops, embeddings, epochs,
                                stats, detection_cost, adj availability
GET  /api/buffer?file=<pts>  -> packed binary buffer for GPU upload (cached,
                                Range-serveable so the client streams it in chunks)
GET  /api/node?file=&idxs=      -> full metadata for a few buffer rows (on selection)
GET  /api/search?file=&q=       -> buffer rows matching a query (id / path / cmd)
POST /api/filter {file,terms}   -> buffer rows matching any of many terms (CSV filter)
GET  /api/neighbors?file=&node= -> one node's incident edges (See Edges)
GET  /api/causal?file=&node=    -> causal subgraph (server-side trace)
GET  /api/attack_pairs?file=    -> malicious<->malicious edge pairs
GET  /api/campaign?file=        -> campaign attack graph
(Adjacency is served from a persisted CSR index, not the raw adj.json, and is
 warmed in the background when a run is opened.)

Binary buffer layout (little-endian), with n points and H hops, in order:
  1. positions : float16[H * n * 3]   (hop-major, then point, then x,y,z)
  2. attrs     : uint16[n * 4]        (tw_idx, tw_start, tw_end, score-as-float16)
  3. ids       : uint32[n]            (node_id)
  4. meta      : uint8[n]             (bit-packed: label<<0 | det<<1 | type<<3)
  (no colour column, and no size — the client derives both from the packed flags)

Run::  python -m pidsmaker.vizgen.web.viz_server  [--host H] [--port P]
"""

import argparse
import glob
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
PIDSMAKER_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def get_artifacts_root():
    """Mirror loader.resolve_latest_viz_dir's root resolution."""
    if os.path.isdir("/home/artifacts"):
        return "/home/artifacts"
    return os.environ.get(
        "PIDS_ARTIFACTS_DIR", os.path.join(PIDSMAKER_ROOT, "artifacts")
    )


ARTIFACTS_ROOT = get_artifacts_root()


def _type_enum(t):
    t = (t or "").lower()
    if "process" in t or "subject" in t:
        return 0
    if "file" in t:
        return 1
    if "netflow" in t:
        return 2
    return 3


# --------------------------------------------------------------------------- #
# Security: only allow files inside the artifacts root
# --------------------------------------------------------------------------- #

def safe_path(p, kind="file"):
    """Resolve *p* and ensure it lives under ARTIFACTS_ROOT. Abort otherwise.

    kind="file" requires an existing file; kind="dir" an existing directory.
    """
    if not p:
        abort(400, "missing path param")
    real = os.path.realpath(p)
    root = os.path.realpath(ARTIFACTS_ROOT)
    if not (real == root or real.startswith(root + os.sep)):
        abort(403, "path outside artifacts root")
    if kind == "dir":
        if not os.path.isdir(real):
            abort(404, "directory not found")
    elif not os.path.isfile(real):
        abort(404, "file not found")
    return real


# --------------------------------------------------------------------------- #
# Run discovery (replicates run_browser.py scan)
# --------------------------------------------------------------------------- #

def _read_model_name(config_path, viz_dir):
    model = "Unknown"
    try:
        if config_path and os.path.exists(config_path):
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("_model", "Unknown") or "Unknown"
    except Exception:
        pass
    if model == "Unknown" and viz_dir and os.path.isdir(viz_dir):
        files = os.listdir(viz_dir)
        if any("word2vec" in f for f in files):
            model = "Velox (Word2Vec)"
        elif any("encoder" in f for f in files):
            model = "Orthrus (GNN)"
    return model


def _raw_model(cfg_path):
    """Read the raw `_model` key from a run_config.yml (lowercased), or ''."""
    try:
        if cfg_path and os.path.exists(cfg_path):
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            return str(cfg.get("_model", "") or "").lower()
    except Exception:
        pass
    return ""


def _featurization_method(cfg_path):
    """The run's node featurization method (e.g. word2vec, doc2vec, flash, fasttext).

    The base ("non-encoder") embedding space is whatever featurization the run
    used — not necessarily word2vec — so the UI labels it with this instead of a
    hardcoded "Word2Vec". Returns '' if it can't be determined.
    """
    try:
        if cfg_path and os.path.exists(cfg_path):
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            feat = cfg.get("featurization")
            if isinstance(feat, dict):
                m = feat.get("used_method") or feat.get("used_methods")
                if isinstance(m, str) and m:
                    return m
    except Exception:
        pass
    return ""


def discover_runs():
    """Scan artifacts for evaluation runs.

    A run is listed if it has a viz_manifest.json (evaluated) and/or already has
    viz/*_points.json. Each entry carries a status so the UI can offer
    generation: ``ready`` (expected embeddings present), ``partial`` (some
    present), or ``needs_viz`` (manifest only, nothing generated yet).
    """
    runs = []
    seen = set()
    patterns = [
        os.path.join(ARTIFACTS_ROOT, "evaluation/evaluation/*/*"),
        os.path.join(ARTIFACTS_ROOT, "detection/evaluation/*/*"),
    ]
    for pat in patterns:
        for d_path in glob.glob(pat):
            if not os.path.isdir(d_path) or d_path in seen:
                continue
            viz_dir = os.path.join(d_path, "viz")
            manifest_path = os.path.join(d_path, "viz_manifest.json")
            points = sorted(glob.glob(os.path.join(viz_dir, "*_points.json"))) \
                if os.path.isdir(viz_dir) else []
            has_manifest = os.path.exists(manifest_path)
            if not points and not has_manifest:
                continue
            seen.add(d_path)

            dataset = os.path.basename(d_path)
            full_hash = os.path.basename(os.path.dirname(d_path))
            cfg_path = os.path.join(d_path, "run_config.yml")
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(os.path.dirname(d_path), "run_config.yml")

            manifest = load_manifest(viz_dir)
            best_epoch = None
            epochs = []
            if manifest and manifest.get("epochs"):
                eps = manifest["epochs"]
                best_epoch = sorted(eps, key=lambda e: e.get("adp", 0), reverse=True)[0].get("epoch")
                epochs = [e.get("epoch") for e in eps]

            basenames = [os.path.basename(p) for p in points]
            present_w2v = any("word2vec" in b for b in basenames)
            present_enc = any("encoder" in b for b in basenames)

            # expected embeddings from the model type
            model_lc = _raw_model(cfg_path)
            if "velox" in model_lc:
                expected = {"w2v"}
            elif any(k in model_lc for k in ("orthrus", "rcaid", "gnn")):
                expected = {"enc"}
            else:
                expected = set()
            present = set()
            if present_w2v:
                present.add("w2v")
            if present_enc:
                present.add("enc")
            if not present:
                status = "needs_viz"
            elif expected and not expected.issubset(present):
                status = "partial"
            else:
                status = "ready"

            # default file to open: bare encoder > best-epoch encoder > word2vec > first
            default_file = None
            for fp in points:
                b = os.path.basename(fp)
                if "encoder" in b and "epoch" not in b:
                    default_file = fp
                    break
            if default_file is None and best_epoch is not None:
                for fp in points:
                    if f"encoder_epoch_{best_epoch}_" in os.path.basename(fp):
                        default_file = fp
                        break
            if default_file is None and points:
                w2v = [fp for fp in points if "word2vec" in os.path.basename(fp)]
                default_file = w2v[0] if w2v else points[0]

            adp, disc = 0.0, 0.0
            if manifest and manifest.get("epochs"):
                best = sorted(manifest["epochs"], key=lambda e: e.get("adp", 0), reverse=True)[0]
                adp, disc = best.get("adp", 0.0), best.get("disc_score", 0.0)

            mtime = os.path.getmtime(d_path)
            runs.append(
                {
                    "dataset": dataset,
                    "hash": full_hash[:8],
                    "full_hash": full_hash,
                    "model": _read_model_name(cfg_path, viz_dir),
                    "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)),
                    "mtime": mtime,
                    "viz_dir": viz_dir,
                    "eval_dir": d_path,
                    "n_files": len(points),
                    "default_file": default_file,
                    "status": status,
                    "present": {"word2vec": present_w2v, "encoder": present_enc},
                    "epochs": epochs,
                    "adp": adp,
                    "disc_score": disc,
                    "has_manifest": has_manifest,
                }
            )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _label_for_file(basename, best_epoch=None, featurization=None):
    if "word2vec" in basename:
        return f"Featurization ({featurization})" if featurization else "Featurization Embedding"
    if "encoder_epoch_" in basename:
        ep = basename.split("encoder_epoch_")[1].split("_")[0]
        return f"GNN Encoder (Epoch {ep})"
    if "encoder" in basename:
        return (
            f"GNN Encoder (Best: Epoch {best_epoch})"
            if best_epoch is not None
            else "GNN Encoder (Best)"
        )
    return basename


def _epoch_of(basename):
    if "encoder_epoch_" in basename:
        try:
            return int(basename.split("encoder_epoch_")[1].split("_")[0])
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# run_config.yml cleaning (parity with loader.load_data)
# --------------------------------------------------------------------------- #

KNOWN_METHODS = {
    "alacarte", "doc2vec", "fasttext", "flash", "temporal_rw", "word2vec",
    "custom_mlp", "gat", "gin", "graph_attention", "magic_gat", "sage", "tgn",
    "none", "rcaid_gat", "sum_aggregation", "glstm", "few_shot",
    "predict_edge_contrastive", "predict_edge_type", "predict_node_type",
    "reconstruct_edge_embeddings", "reconstruct_node_embeddings",
    "reconstruct_node_features", "reconstruct_masked_features",
    "predict_masked_struct", "detect_edge_few_shot", "global_batching",
    "inter_graph_batching", "intra_graph_batching", "edges",
    "tgn_last_neighbor", "depimpact", "synthetic_attack_naive",
    "rcaid_pseudo_graph", "kairos_idf_queue", "provnet_lof_queue",
}


def _clean_cfg(d):
    if not isinstance(d, dict):
        return d
    active = None
    if isinstance(d.get("used_method"), str):
        active = d["used_method"]
    elif isinstance(d.get("used_methods"), str):
        active = d["used_methods"]
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        if v is None or v == "" or v == [] or v == {}:
            continue
        if k in {
            "attack_to_time_window", "ground_truth_relative_path", "train_dates",
            "test_dates", "val_dates", "unused_dates", "database",
            "database_all_file", "host", "password", "port", "user",
            "node_label_features",
        }:
            continue
        if active and isinstance(v, dict) and k in KNOWN_METHODS and k != active:
            continue
        if isinstance(v, dict):
            vc = _clean_cfg(v)
            if vc:
                if (
                    len(vc) == 1
                    and list(vc.keys())[0] in ("used_method", "used_methods")
                    and vc[list(vc.keys())[0]] == "none"
                ):
                    continue
                out[k] = vc
        else:
            out[k] = v
    return out


def load_run_config_text(viz_dir):
    eval_dir = os.path.dirname(viz_dir)
    for cand in (
        os.path.join(eval_dir, "run_config.yml"),
        os.path.join(os.path.dirname(eval_dir), "run_config.yml"),
    ):
        if os.path.exists(cand):
            try:
                import yaml

                with open(cand) as f:
                    cfg = yaml.safe_load(f)
                return yaml.dump(_clean_cfg(cfg), default_flow_style=False, sort_keys=True)
            except Exception:
                return ""
    return ""


# --------------------------------------------------------------------------- #
# Manifest helpers (adp / disc_score)
# --------------------------------------------------------------------------- #

def load_manifest(viz_dir):
    mf = os.path.join(os.path.dirname(viz_dir), "viz_manifest.json")
    if os.path.exists(mf):
        try:
            with open(mf) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def metrics_for_file(basename, manifest):
    """Return (adp, disc_score, best_epoch) for a points file."""
    if manifest is None or "word2vec" in basename:
        return 0.0, 0.0, None
    epochs = manifest.get("epochs", [])
    if not epochs:
        return 0.0, 0.0, None
    best = sorted(epochs, key=lambda e: e.get("adp", 0), reverse=True)[0]
    best_ep = best.get("epoch")
    ep = _epoch_of(basename)
    if ep is not None:
        for e in epochs:
            if str(e.get("epoch")) == str(ep):
                return e.get("adp", 0.0), e.get("disc_score", 0.0), best_ep
    return best.get("adp", 0.0), best.get("disc_score", 0.0), best_ep


# --------------------------------------------------------------------------- #
# Core: parse points.json -> numpy arrays (with disk cache)
# --------------------------------------------------------------------------- #

CACHE_VERSION = 4   # v4: v3 + attrs packed (tw uint16 ×3 + score float16, size dropped)


def _cache_paths(points_path):
    base = points_path + f".webcache_v{CACHE_VERSION}"
    return base + ".bin", base + ".meta.json", base + ".info.json"


def _pack_attrs(attrs):
    """attrs f32[n,5] (tw_idx, tw_start, tw_end, score, size) -> uint16[n,4]
    (tw_idx, tw_start, tw_end, score-as-float16). `size` is dropped (the client
    derives it from the label); the temporal 'never ends' sentinel (1e30) clips
    to 65535, which the client reads back as +inf."""
    attrs = np.asarray(attrs, dtype=np.float32)
    n = len(attrs)
    packed = np.empty((n, 4), np.uint16)
    packed[:, 0] = np.clip(attrs[:, 0], 0, 65535).astype(np.uint16)
    packed[:, 1] = np.clip(attrs[:, 1], 0, 65535).astype(np.uint16)
    packed[:, 2] = np.clip(attrs[:, 2], 0, 65535).astype(np.uint16)
    packed[:, 3] = attrs[:, 3].astype(np.float16).view(np.uint16)
    return packed


def _cache_fresh(points_path):
    bin_p, meta_p, info_p = _cache_paths(points_path)
    if not (os.path.exists(bin_p) and os.path.exists(meta_p) and os.path.exists(info_p)):
        return False
    src_m = os.path.getmtime(points_path)
    return all(os.path.getmtime(p) >= src_m for p in (bin_p, meta_p, info_p))


def build_cache(points_path):
    """Parse points.json and build the binary cache from it (server path)."""
    try:
        with open(points_path) as f:
            pts = json.load(f)
    except (json.JSONDecodeError, ValueError):
        abort(422, "points file is corrupt or incomplete; regenerate this run")
    if not pts:
        abort(422, "empty points file")
    return build_point_cache(pts, points_path)


def build_point_cache(pts, points_path):
    """Build the .bin/.meta/.info cache from in-memory point records (the same
    shape as points.json). Shared by the server (which parses points.json) and
    the exporter (which passes its records directly), so both produce identical
    caches and the exporter's output needs no server-side JSON re-parse."""
    t0 = time.time()
    n = len(pts)

    first = pts[0]
    H = len(first["coords_hops"]) if "coords_hops" in first else 1

    pos = np.zeros((H, n, 3), dtype=np.float32)
    attrs = np.zeros((n, 5), dtype=np.float32)  # tw_idx, tw_start, tw_end, score, size
    flags = np.zeros((n, 4), dtype=np.uint32)   # node_id, label, det, type_enum
    # NOTE: no colour column. The client derives every node's colour from the flags
    # (label/detection/type) via nodeRgba, so shipping colours was ~25% of the
    # buffer wasted — dropped from the v2 cache.

    # metadata (columnar) for inspector / search / tables
    ids = [0] * n
    paths = [""] * n
    types = [""] * n
    scores = [0.0] * n
    tw_labels = [""] * n
    cmds = [""] * n
    top_edges = [""] * n

    node_tws = {}  # node_id -> list of (tw_idx, array_index)

    stats = {
        "total": n, "benign": 0, "malicious": 0,
        "mal_proc": 0, "mal_file": 0, "mal_net": 0,
        "attack_start_tw": math.inf, "attack_start_time": "",
    }

    for i, p in enumerate(pts):
        if "coords_hops" in p:
            ch = p["coords_hops"]
            for h in range(H):
                c = ch[h] if h < len(ch) else ch[-1]
                pos[h, i, 0] = c[0]
                pos[h, i, 1] = c[1]
                pos[h, i, 2] = c[2]
        else:
            pos[0, i] = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]

        lbl = int(p.get("label", 0) or 0)
        det = int(p.get("detection_status", 0) or 0)
        tw_idx = float(p.get("tw_idx", 0) or 0)
        score = float(p.get("anomaly_score", 0.0) or 0.0)
        ptype = p.get("type") or ""
        nid = int(p.get("node_id", 0) or 0)

        attrs[i, 0] = tw_idx
        attrs[i, 3] = score
        if lbl == 0:
            stats["benign"] += 1
            attrs[i, 4] = 3.0
        else:
            stats["malicious"] += 1
            attrs[i, 4] = 5.0
            if tw_idx < stats["attack_start_tw"]:
                stats["attack_start_tw"] = tw_idx
                stats["attack_start_time"] = p.get("tw_label", "")
            te = _type_enum(ptype)
            if te == 0:
                stats["mal_proc"] += 1
            elif te == 1:
                stats["mal_file"] += 1
            elif te == 2:
                stats["mal_net"] += 1

        flags[i] = [nid, lbl, det, _type_enum(ptype)]

        ids[i] = nid
        paths[i] = p.get("path", "")
        types[i] = ptype
        scores[i] = score
        tw_labels[i] = p.get("tw_label", "")
        cmds[i] = p.get("cmd", "")
        top_edges[i] = p.get("top_edge", "")

        node_tws.setdefault(nid, []).append((tw_idx, i))

    # temporal lifespan: tw_start = tw_idx ; tw_end = next occurrence of same id
    attrs[:, 1] = attrs[:, 0]
    attrs[:, 2] = np.inf
    for lst in node_tws.values():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: x[0])
        for k in range(len(lst) - 1):
            attrs[lst[k][1], 2] = lst[k + 1][0]
    # encode inf -> very large finite (shader compares u_time >= a_tw_end)
    attrs[np.isinf(attrs[:, 2]), 2] = 1e30

    max_tw = int(np.max(attrs[:, 0])) if n else 0
    if math.isinf(stats["attack_start_tw"]):
        stats["attack_start_tw"] = -1

    # No index reordering: the exporter's natural node order is kept, so the
    # painter's draw order matches the original (depth-correct-looking) layout.
    # The client renders from the binary buffer and loads the string metadata
    # lazily, so an "attacks-first" prefix is no longer needed.

    # --- write binary buffer (atomically: temp file + rename) ---
    bin_p, meta_p, info_p = _cache_paths(points_path)
    bin_tmp = bin_p + ".tmp"
    # Pack the buffer to cut wire transfer: positions as float16 (UMAP coords need
    # nowhere near f32 precision), flags as node_id(u32) + one meta byte holding
    # label(1b) | det(2b) | type(2b). Colour is already gone (derived from flags).
    node_ids = flags[:, 0].astype(np.uint32)
    meta_byte = (flags[:, 1] | (flags[:, 2] << 1) | (flags[:, 3] << 3)).astype(np.uint8)
    with open(bin_tmp, "wb") as f:
        f.write(np.ascontiguousarray(pos.astype(np.float16)).tobytes())  # H*n*3 f16
        f.write(np.ascontiguousarray(_pack_attrs(attrs)).tobytes())      # n*4 u16
        f.write(np.ascontiguousarray(node_ids).tobytes())                # n u32
        f.write(np.ascontiguousarray(meta_byte).tobytes())               # n u8
    os.replace(bin_tmp, bin_p)

    # --- campaign mapping (optional) ---
    campaign_ids = None
    num_campaigns = 0
    camp_path = os.path.join(os.path.dirname(points_path), "campaign_mapping.json")
    if os.path.exists(camp_path):
        try:
            with open(camp_path) as f:
                cm = json.load(f)
            n2a = cm.get("node2attacks", {})
            campaign_ids = [n2a.get(str(ids[i]), []) for i in range(n)]
            num_campaigns = cm.get("num_campaigns", 0)
        except Exception:
            campaign_ids = None

    meta = {
        "ids": ids, "paths": paths, "types": types, "scores": scores,
        "tw_idx": attrs[:, 0].astype(int).tolist(), "tw_labels": tw_labels,
        "cmds": cmds, "top_edges": top_edges,
        "labels": flags[:, 1].astype(int).tolist(),
        "det": flags[:, 2].astype(int).tolist(),
    }
    if campaign_ids is not None:
        meta["campaigns"] = campaign_ids
    meta_tmp = meta_p + ".tmp"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    os.replace(meta_tmp, meta_p)
    build_meta_store(meta, meta_p + ".store.npz")  # binary store -> instant metadata

    info = {
        "n": n, "hops": H, "max_tw": max_tw,
        "stats": stats, "num_campaigns": num_campaigns,
        "byte_offsets": {           # layout: positions(f16) | attrs(u16×4) | ids(u32) | meta(u8)
            "positions": 0,
            "attrs": H * n * 3 * 2,
            "ids": H * n * 3 * 2 + n * 4 * 2,
            "meta": H * n * 3 * 2 + n * 4 * 2 + n * 4,
        },
    }
    # detection cost needs scores + labels + det + campaigns
    info["detection_cost"] = compute_detection_cost(
        np.array(scores), np.array(meta["labels"]), np.array(meta["det"]),
        np.array(ids), meta.get("campaigns"), num_campaigns,
    )
    info_tmp = info_p + ".tmp"
    with open(info_tmp, "w") as f:
        json.dump(info, f)
    os.replace(info_tmp, info_p)

    print(f"[viz_server] cached {os.path.basename(points_path)} "
          f"({n} pts, {H} hops) in {time.time()-t0:.1f}s")
    return info


# --------------------------------------------------------------------------- #
# Detection cost sweep (parity with main_window.precompute_detection_cost)
# --------------------------------------------------------------------------- #

def compute_detection_cost(scores, labels, det, ids, campaigns, num_campaigns):
    n = len(scores)
    if n == 0:
        return None
    benign = labels == 0
    mal = labels == 1

    total_gt = len(np.unique(ids[mal])) if mal.any() else 0
    detected_ids = np.unique(ids[(mal) & (det == 1)]) if mal.any() else np.array([])
    detected = len(detected_ids)

    out = {
        "total_gt": int(total_gt),
        "detected": int(detected),
        "current_fp": None, "current_threshold": None,
        "fp_full_recall": None, "thresh_full_recall": None,
        "fp_full_campaign": None, "thresh_full_campaign": None,
        "campaign_coverage_det": None, "num_campaigns": int(num_campaigns or 0),
    }

    det_scores = scores[(mal) & (det == 1)]
    if det_scores.size:
        cur_thr = float(np.min(det_scores))
        out["current_threshold"] = cur_thr
        out["current_fp"] = int(np.sum(benign & (scores >= cur_thr)))

    # FP for 100% recall: lowest mal score must pass
    if mal.any():
        thr_recall = float(np.min(scores[mal]))
        out["thresh_full_recall"] = thr_recall
        out["fp_full_recall"] = int(np.sum(benign & (scores >= thr_recall)))

    # campaign coverage
    if campaigns and num_campaigns:
        det_camps = set()
        for i in np.where((mal) & (det == 1))[0]:
            for c in campaigns[i]:
                det_camps.add(c)
        out["campaign_coverage_det"] = len(det_camps)
        # threshold so at least one node per campaign passes
        camp_best = {}
        for i in np.where(mal)[0]:
            for c in campaigns[i]:
                camp_best[c] = max(camp_best.get(c, -1e30), scores[i])
        if camp_best:
            thr_camp = float(min(camp_best.values()))
            out["thresh_full_campaign"] = thr_camp
            out["fp_full_campaign"] = int(np.sum(benign & (scores >= thr_camp)))
    return out


_build_locks = {}
_build_locks_guard = threading.Lock()


def _build_lock(path):
    with _build_locks_guard:
        lk = _build_locks.get(path)
        if lk is None:
            lk = _build_locks[path] = threading.Lock()
        return lk


def _read_info(points_path):
    _, _, info_p = _cache_paths(points_path)
    with open(info_p) as f:
        return json.load(f)


def _migrate_old_cache(points_path):
    """Upgrade an existing v1/v2/v3 cache to the current buffer format by
    transforming the binary directly (float16 positions, dropped colour, packed
    flags, packed attrs) instead of re-parsing the 100s-of-MB points.json — ~1 s
    vs ~30 s+. The metadata store is format-unchanged, so it is just copied.
    Returns the v4 info dict, or None if there is no usable old cache to migrate."""
    src_m = os.path.getmtime(points_path) if os.path.exists(points_path) else 0
    bin_p, meta_p, info_p = _cache_paths(points_path)                 # v4 targets
    for old_v in (3, 2, 1):
        base = points_path + f".webcache_v{old_v}"
        o_bin, o_meta, o_info, o_store = base + ".bin", base + ".meta.json", \
            base + ".info.json", base + ".meta.json.store.npz"
        if not all(os.path.exists(p) for p in (o_bin, o_meta, o_info, o_store)):
            continue
        if os.path.getmtime(o_bin) < src_m:      # stale vs source -> real rebuild
            continue
        try:
            with open(o_info) as f:
                info = json.load(f)
            n, H, bo = info["n"], info["hops"], info["byte_offsets"]
            raw = np.fromfile(o_bin, dtype=np.uint8)
            attrs = raw[bo["attrs"]:bo["attrs"] + n * 5 * 4].view(np.float32).reshape(n, 5)
            if old_v == 3:   # already f16 pos + u32 ids + u8 meta; only attrs repack
                pos_bytes = raw[bo["positions"]:bo["positions"] + H * n * 3 * 2].tobytes()
                ids_bytes = raw[bo["ids"]:bo["ids"] + n * 4].tobytes()
                meta_bytes = raw[bo["meta"]:bo["meta"] + n].tobytes()
            else:            # v1/v2: f32 pos, u32×4 flags -> f16 pos, id + meta byte
                pos = raw[bo["positions"]:bo["positions"] + H * n * 3 * 4].view(np.float32)
                flags = raw[bo["flags"]:bo["flags"] + n * 4 * 4].view(np.uint32).reshape(n, 4)
                pos_bytes = pos.astype(np.float16).tobytes()
                ids_bytes = flags[:, 0].astype(np.uint32).tobytes()
                meta_bytes = (flags[:, 1] | (flags[:, 2] << 1) | (flags[:, 3] << 3)).astype(np.uint8).tobytes()
            tmp = bin_p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(pos_bytes)
                f.write(np.ascontiguousarray(_pack_attrs(attrs)).tobytes())
                f.write(ids_bytes)
                f.write(meta_bytes)
            os.replace(tmp, bin_p)
            shutil.copyfile(o_store, meta_p + ".store.npz")
            shutil.copyfile(o_meta, meta_p)
            info["byte_offsets"] = {"positions": 0, "attrs": H * n * 3 * 2,
                                    "ids": H * n * 3 * 2 + n * 4 * 2,
                                    "meta": H * n * 3 * 2 + n * 4 * 2 + n * 4}
            with open(info_p + ".tmp", "w") as f:
                json.dump(info, f)
            os.replace(info_p + ".tmp", info_p)
            print(f"[viz_server] migrated {os.path.basename(points_path)} v{old_v}->v{CACHE_VERSION} (binary, no reparse)")
            return info
        except Exception as e:
            print(f"[viz_server] cache migration failed ({type(e).__name__}: {e}); full rebuild")
            return None
    return None


def get_info(points_path):
    """Return cached run info, building the cache once if stale.

    A per-file lock makes the build single-flight: concurrent requests for the
    same cold run wait for one build instead of racing (which could publish a
    half-written cache). build_cache writes atomically, so a fresh cache is
    always complete. If only an older-version cache exists, migrate its binary
    in place (fast) rather than re-parsing points.json.
    """
    if _cache_fresh(points_path):
        return _read_info(points_path)
    with _build_lock(points_path):
        if _cache_fresh(points_path):  # built while we waited on the lock
            return _read_info(points_path)
        migrated = _migrate_old_cache(points_path)   # fast v1/v2 -> v3 binary upgrade
        if migrated is not None:
            return migrated
        return build_cache(points_path)


# --------------------------------------------------------------------------- #
# Adjacency index (server-side).
#
# The adjacency (``*_adj.json``) is large (tens to hundreds of MB) and re-parsing
# it on every "See Edges" / causal-trace / attack-overlay action does not scale.
# Instead we build a compact CSR index once and persist it next to the run as a
# binary ``.npz``: parsing 75 MB of JSON takes ~2.4 s, but the CSR reloads in
# ~0.02 s and is half the size — so the cost is paid once, ever (it survives
# server restarts), and every later query is a small array slice. The index is
# also warmed in a background thread when a run is opened (see api_run), so the
# first click is instant. Single-entry in-memory cache bounds RAM to one run.
# --------------------------------------------------------------------------- #

ADJ_INDEX_VERSION = 1
_META_CACHE = {}
_ADJ_INDEX = {}


class AdjIndex:
    """CSR adjacency: ``ids[K]`` sorted node ids, ``indptr[K+1]`` offsets, and
    one entry per directed edge — ``nb[M]`` neighbour, ``t[M]`` time,
    ``dir[M]`` (1=out, 0=in), ``et[M]`` code into ``vocab`` (relation names)."""

    def __init__(self, ids, indptr, nb, t, dr, et, vocab):
        self.ids, self.indptr = ids, indptr
        self.nb, self.t, self.dir, self.et = nb, t, dr, et
        self.vocab = vocab

    def _row(self, node):
        i = int(np.searchsorted(self.ids, node))
        if 0 <= i < len(self.ids) and int(self.ids[i]) == int(node):
            return i
        return -1

    def neighbors(self, node):
        """This node's incident edges as ``[{nb, t, dir, et?}]`` — the same shape
        the old per-node adjacency slice returned."""
        i = self._row(node)
        if i < 0:
            return []
        s, e = int(self.indptr[i]), int(self.indptr[i + 1])
        out = []
        for j in range(s, e):
            d = {"nb": int(self.nb[j]), "t": int(self.t[j]),
                 "dir": "out" if self.dir[j] else "in"}
            c = int(self.et[j])
            if 0 <= c < len(self.vocab) and self.vocab[c]:
                d["et"] = self.vocab[c]
            out.append(d)
        return out


def _adj_index_paths(points_path):
    adj_path = points_path.replace("_points.json", "_adj.json")
    base = f"{adj_path}.idxcache_v{ADJ_INDEX_VERSION}"
    return adj_path, base + ".npz", base + ".vocab.json"


def _adj_index_fresh(adj_path, npz_path, vocab_path):
    if not (os.path.exists(npz_path) and os.path.exists(vocab_path)):
        return False
    src = os.path.getmtime(adj_path)
    return all(os.path.getmtime(p) >= src for p in (npz_path, vocab_path))


def _build_adj_index(adj_path, npz_path, vocab_path):
    """Parse adj.json once and build the CSR index from it (server path)."""
    with open(adj_path) as f:
        adj = json.load(f)
    return build_adj_index_from_dict(adj, adj_path, npz_path, vocab_path)


def build_adj_index_from_dict(adj, adj_path, npz_path, vocab_path):
    """Write the CSR ``.npz`` + relation vocab from an in-memory adjacency dict
    ({node_id -> [{nb, t, dir, et?}]}). Shared by the server (parses adj.json)
    and the exporter (passes its dict directly), so no server-side re-parse."""
    t0 = time.time()
    keys = list(adj.keys())
    ids = np.fromiter((int(k) for k in keys), dtype=np.int64, count=len(keys))
    order = np.argsort(ids)
    deg = np.fromiter((len(adj[keys[o]]) for o in order), dtype=np.int64, count=len(keys))
    M = int(deg.sum())
    indptr = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    nb = np.empty(M, np.int64)
    t = np.empty(M, np.int32)
    dr = np.empty(M, np.uint8)
    et = np.empty(M, np.int32)
    vocab, vidx = [""], {"": 0}
    p = 0
    for o in order:
        for e in adj[keys[o]]:
            nb[p] = e["nb"]
            t[p] = e.get("t", 0)
            dr[p] = 1 if e.get("dir") == "out" else 0
            s = e.get("et") or ""
            c = vidx.get(s)
            if c is None:
                c = vidx[s] = len(vocab)
                vocab.append(s)
            et[p] = c
            p += 1
    ids_sorted = ids[order]
    tmp = npz_path + ".tmp.npz"
    np.savez(tmp, ids=ids_sorted, indptr=indptr, nb=nb, t=t, dir=dr, et=et)
    os.replace(tmp if os.path.exists(tmp) else tmp + ".npz", npz_path)
    vtmp = vocab_path + ".tmp"
    with open(vtmp, "w") as f:
        json.dump(vocab, f)
    os.replace(vtmp, vocab_path)
    print(f"[viz_server] built adjacency index for {os.path.basename(adj_path)} "
          f"({len(keys)} nodes, {M} edges) in {time.time()-t0:.1f}s")
    return AdjIndex(ids_sorted, indptr, nb, t, dr, et, vocab)


def _load_adj_index(npz_path, vocab_path):
    z = np.load(npz_path)
    with open(vocab_path) as f:
        vocab = json.load(f)
    return AdjIndex(z["ids"], z["indptr"], z["nb"], z["t"], z["dir"], z["et"], vocab)


def get_adj_index(points_path):
    """Return the run's :class:`AdjIndex`, building+persisting it on first use.

    Single-flight (per-file lock) and cached in memory; returns None if the run
    has no adjacency file.
    """
    adj_path, npz_path, vocab_path = _adj_index_paths(points_path)
    if not os.path.exists(adj_path):
        return None
    key = (adj_path, os.path.getmtime(adj_path))
    if _ADJ_INDEX.get("key") == key:
        return _ADJ_INDEX["index"]
    with _build_lock(adj_path):
        if _ADJ_INDEX.get("key") == key:  # built while we waited
            return _ADJ_INDEX["index"]
        idx = None
        if _adj_index_fresh(adj_path, npz_path, vocab_path):
            try:
                idx = _load_adj_index(npz_path, vocab_path)
            except Exception as e:  # corrupt/partial/version-skewed cache -> rebuild
                print(f"[viz_server] adj index reload failed ({e}); rebuilding")
        if idx is None:
            idx = _build_adj_index(adj_path, npz_path, vocab_path)
        _ADJ_INDEX.clear()
        _ADJ_INDEX["key"] = key
        _ADJ_INDEX["index"] = idx
        return idx


# Heavy background cache builds are serialized through this semaphore so warming
# a run's many sibling epoch caches uses ~one core instead of thrashing all of
# them at once (and competing with whatever the user is actively viewing).
_WARM_SEMA = threading.Semaphore(1)


def warm_run(points_path):
    """Warm everything needed to make a freshly-opened run feel instant.

    First the opened file's graph tools (adjacency index, derived id-maps), then
    — low priority, one at a time — the point caches of the sibling embedding /
    epoch files, so switching embedding space or epoch does not pay a cold
    ``build_cache`` parse of a 100-600 MB points file. Best-effort.

    NOT the campaign graph: it is a pure-Python build that, on a large run, holds
    the GIL long enough to starve the buffer/meta requests the user is actively
    waiting on. It is cached on first click of "Campaign Attack Graph" instead.
    """
    try:
        get_adj_index(points_path)
        load_meta(points_path)      # builds/loads the binary metadata store
    except Exception as e:
        print(f"[viz_server] warm_run(graph) failed for {os.path.basename(points_path)}: {e}")

    try:
        viz_dir = os.path.dirname(points_path)
        siblings = [s for s in sorted(glob.glob(os.path.join(viz_dir, "*_points.json")))
                    if s != points_path]
    except Exception:
        siblings = []
    for sib in siblings:
        if _cache_fresh(sib):
            continue
        with _WARM_SEMA:  # one heavy build at a time, globally
            try:
                if not _cache_fresh(sib):
                    get_info(sib)  # single-flight build; a concurrent switch shares it
                    print(f"[viz_server] warmed switch cache: {os.path.basename(sib)}")
            except Exception as e:
                print(f"[viz_server] warm sibling cache failed for {os.path.basename(sib)}: {e}")


# --------------------------------------------------------------------------- #
# Binary metadata store.
#
# The columnar metadata is mostly strings (paths); parsing the JSON and
# materialising millions of Python strings is the dominant open cost at scale
# (~10 s for THEIA's 2.1 M rows, worse for E5). Instead we persist a binary
# store: numeric columns are plain arrays (instant load), string columns are one
# byte-blob + offsets (decode only the rows actually requested), and a sorted
# unique-id index gives O(log n) id->row lookup with no per-request scan. Built
# once (at export, or lazily from meta.json for older runs) and reused forever.
# --------------------------------------------------------------------------- #

_meta_lock = threading.Lock()
_META_STRCOLS = ("paths", "types", "cmds", "tw_labels", "top_edges")

_OFF_GRAPH = {"tw": "—", "type": "Filtered/Off-Graph", "score": 0,
              "path": "", "cmd": "", "label": 0, "twi": None}


class StrCol:
    """A string column stored as a UTF-8 blob + int64 offsets. Decodes lazily,
    so a slice/lookup costs only the strings it touches, not the whole column."""

    def __init__(self, blob, off):
        self.blob, self.off = blob, off
        self._raw = None

    def raw_bytes(self):
        """The whole column as one bytes object (cached) — for a full-text scan
        that never decodes individual rows."""
        if self._raw is None:
            self._raw = self.blob.tobytes()
        return self._raw

    def __getitem__(self, i):
        """Decode a single row (used by the per-node inspector / See-Edges rows)."""
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.blob[a:b].tobytes().decode("utf-8", "replace") if b > a else ""


class MetaStore:
    """Columnar metadata backed by binary arrays; dict-style access for numeric
    columns returns numpy arrays, string columns return lazy :class:`StrCol`."""

    def __init__(self, z):
        self._num = {k: z[k] for k in ("ids", "scores", "labels", "det", "tw_idx", "type_enum")}
        self._str = {c: StrCol(z[c + "_blob"], z[c + "_off"]) for c in _META_STRCOLS}
        self.n = len(self._num["ids"])
        self.uniq_ids = z["uniq_ids"]                     # sorted unique node ids
        self.first_idx = z["first_idx"]                   # first row index per unique id
        self.first_tw = self._num["tw_idx"][self.first_idx]
        self.first_label = self._num["labels"][self.first_idx]
        self._camp = (z["camp_data"], z["camp_off"]) if "camp_data" in z.files else None

    def __contains__(self, k):
        return k in self._num or k in self._str or (k == "campaigns" and self._camp is not None)

    def __getitem__(self, k):
        if k in self._num:
            return self._num[k]
        if k in self._str:
            return self._str[k]
        if k == "campaigns" and self._camp is not None:
            return self._camp_rows(0, self.n)
        raise KeyError(k)

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def _pos(self, node_id):
        i = int(np.searchsorted(self.uniq_ids, node_id))
        if 0 <= i < len(self.uniq_ids) and int(self.uniq_ids[i]) == int(node_id):
            return i
        return -1

    def first_index(self, node_id):
        p = self._pos(node_id)
        return int(self.first_idx[p]) if p >= 0 else -1

    def first_tw_of(self, node_id):
        p = self._pos(node_id)
        return int(self.first_tw[p]) if p >= 0 else 0

    def label_of(self, node_id):
        p = self._pos(node_id)
        return int(self.first_label[p]) if p >= 0 else 0

    def malicious_ids(self):
        return set(self.uniq_ids[self.first_label != 0].tolist())

    def row(self, i):
        """The per-node fields the inspector tables show (mirrors client rowMeta)."""
        if i < 0:
            return dict(_OFF_GRAPH)
        return {
            "tw": self._str["tw_labels"][i] or int(self._num["tw_idx"][i]),
            "type": self._str["types"][i], "score": float(self._num["scores"][i]),
            "path": self._str["paths"][i], "cmd": self._str["cmds"][i],
            "label": int(self._num["labels"][i]), "twi": int(self._num["tw_idx"][i]),
        }

    def node_at(self, i):
        """Full metadata for buffer row i (== store row i). Used by index-based
        selection so the client needn't hold the node-id column up front. O(1)."""
        if not (0 <= i < self.n):
            return None
        r = self.row(i)
        r["top_edge"] = self._str["top_edges"][i]
        r["id"] = int(self._num["ids"][i])
        return r

    def search(self, terms, limit=20000):
        """Buffer ROWS whose path or command matches any term, plus exact id hits.
        Scans the raw UTF-8 blobs with one alternation regex (no per-row decode),
        maps match byte-offsets back to rows — fast even at 20M+ nodes. Returns
        rows (== buffer indices) so the client filters by index without the ids."""
        if isinstance(terms, str):
            terms = [terms]
        terms = [t.strip().lower() for t in terms if t and t.strip()]
        if not terms:
            return []
        pat = re.compile(b"|".join(re.escape(t.encode("utf-8")) for t in terms), re.IGNORECASE)
        out, seen = [], set()
        for col in (self._str["paths"], self._str["cmds"]):
            blob, off = col.raw_bytes(), col.off
            for m in pat.finditer(blob):
                row = int(np.searchsorted(off, m.start(), side="right")) - 1
                # ignore a match that straddles two concatenated strings
                if 0 <= row < self.n and int(off[row + 1]) >= m.end() and row not in seen:
                    seen.add(row)
                    out.append(row)
                    if len(out) >= limit:
                        return out
        for t in terms:                       # exact node-id match -> all its rows
            if t.isdigit():
                for row in np.nonzero(self._num["ids"] == int(t))[0].tolist():
                    if row not in seen:
                        seen.add(row)
                        out.append(row)
                        if len(out) >= limit:
                            return out
        return out

    def _camp_rows(self, start, end):
        d, o = self._camp
        return [d[int(o[i]):int(o[i + 1])].tolist() for i in range(start, min(end, self.n))]


def _pack_strings(lst):
    enc = [(s.encode("utf-8") if s else b"") for s in lst]
    off = np.zeros(len(enc) + 1, dtype=np.int64)
    np.cumsum(np.fromiter((len(e) for e in enc), dtype=np.int64, count=len(enc)), out=off[1:])
    blob = np.frombuffer(b"".join(enc), dtype=np.uint8) if off[-1] else np.zeros(0, np.uint8)
    return blob, off


def _meta_store_path(points_path):
    _, meta_p, _ = _cache_paths(points_path)
    return meta_p, meta_p + ".store.npz"


def build_meta_store(meta, store_path):
    """Write the binary metadata store from an in-memory columnar meta dict
    (called at export, and lazily from a parsed meta.json for older runs)."""
    ids = np.asarray(meta["ids"], dtype=np.int64)
    arrays = {
        "ids": ids,
        "scores": np.asarray(meta["scores"], dtype=np.float32),
        "labels": np.asarray(meta["labels"], dtype=np.int8),
        "det": np.asarray(meta["det"], dtype=np.int8),
        "tw_idx": np.asarray(meta["tw_idx"], dtype=np.int32),
        "type_enum": np.fromiter((_type_enum(t) for t in meta["types"]), dtype=np.int8, count=len(ids)),
    }
    for c in _META_STRCOLS:
        arrays[c + "_blob"], arrays[c + "_off"] = _pack_strings(meta[c])
    uniq, first = np.unique(ids, return_index=True)
    arrays["uniq_ids"] = uniq
    arrays["first_idx"] = first.astype(np.int64)
    camp = meta.get("campaigns")
    if camp is not None:
        off = np.zeros(len(camp) + 1, dtype=np.int64)
        np.cumsum(np.fromiter((len(c) for c in camp), dtype=np.int64, count=len(camp)), out=off[1:])
        data = np.fromiter((x for c in camp for x in c), dtype=np.int64, count=int(off[-1]))
        arrays["camp_data"], arrays["camp_off"] = data, off
    tmp = store_path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp if os.path.exists(tmp) else tmp + ".npz", store_path)


def load_meta(points_path):
    """Return the run's :class:`MetaStore` (cached, single-flight). Builds the
    binary store once from meta.json if it doesn't exist yet."""
    meta_p, store_p = _meta_store_path(points_path)
    src_m = os.path.getmtime(meta_p) if os.path.exists(meta_p) else 0
    fresh = os.path.exists(store_p) and os.path.getmtime(store_p) >= src_m
    if fresh and _META_CACHE.get("key") == store_p:
        return _META_CACHE["store"]
    with _meta_lock:
        if os.path.exists(store_p) and os.path.getmtime(store_p) >= src_m \
                and _META_CACHE.get("key") == store_p:
            return _META_CACHE["store"]
        if not (os.path.exists(store_p) and os.path.getmtime(store_p) >= src_m):
            with open(meta_p) as f:            # one-time parse for pre-store runs
                build_meta_store(json.load(f), store_p)
        store = MetaStore(np.load(store_p))
        _META_CACHE.clear()
        _META_CACHE["store"] = store
        _META_CACHE["key"] = store_p
        return store


def causal_trace(index, start, start_tw, limit=10000):
    """Time-respecting causal traversal from *start* (mirrors the client BFS).

    Follows out-edges forward in time (t_edge >= t) and in-edges backward
    (t_edge <= t). Returns the ordered list of reached node ids (<= limit).
    """
    visited, result, order = set(), set(), []
    stack = [(start, start_tw)]
    while stack and len(result) < limit:
        node, t = stack.pop()
        k = (node, t)
        if k in visited:
            continue
        visited.add(k)
        if node not in result:
            result.add(node)
            order.append(node)
        for e in index.neighbors(node):
            te = e["t"]
            if e["dir"] == "out" and te >= t:
                stack.append((e["nb"], te))
            elif e["dir"] == "in" and te <= t:
                stack.append((e["nb"], te))
    return order


# --------------------------------------------------------------------------- #
# Export job manager (single concurrent job)
#
# Runs the standalone viz exporter (pidsmaker.vizgen.web.export) as a
# subprocess and streams its stdout to the browser over SSE, so the user can
# generate viz data for a run from the browser and watch the logs live. Spawned
# with PYTHONUNBUFFERED=1 because the exporter's log() uses unflushed print(),
# which would otherwise block-buffer over a pipe.
# --------------------------------------------------------------------------- #

ALLOW_EXPORT = os.environ.get("PIDS_VIZ_ALLOW_EXPORT", "1") not in ("0", "false", "False")

# (key, label, [stdout substrings that mark entering this phase]) — ordered
_PHASES = [
    ("start", "Start", ["embedding extraction..."]),
    ("graphs", "Load graphs", ["Loading preprocessed", "No cached graphs found"]),
    ("model", "Model", ["Loaded model weights", "Evaluation Stats: Found"]),
    ("embeddings", "Embeddings", ["[embed_exporter] Extracted", "Computed ", "Sampled:"]),
    ("umap", "UMAP", ["Running UMAP for Hop", "[dim_reduction] Reducing"]),
    ("write", "Write", ["Building interactive HTML", "Saved visualization to:"]),
]
PHASE_KEYS = [p[0] for p in _PHASES]
PHASE_LABELS = {p[0]: p[1] for p in _PHASES}

_job_lock = threading.Lock()
CURRENT_JOB = None
STARTUP_STALL_LIMIT = 120   # s of silence at the "start" phase before aborting a wedged job


def _classify_phase(line):
    for key, _label, markers in _PHASES:
        if any(m in line for m in markers):
            return key
    return None


def _resolve_model(run_dir, embeddings):
    cfg_path = os.path.join(run_dir, "run_config.yml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(run_dir), "run_config.yml")
    m = _raw_model(cfg_path)
    if m:
        return m
    return "velox" if embeddings == "word2vec" else "orthrus"


def _broadcast(job, msg):
    for q in list(job["subscribers"]):
        try:
            q.put_nowait(msg)
        except Exception:
            pass


def _append_line(job, line):
    """Append a log line with a monotonic index and broadcast it. Shared by the
    subprocess reader and the startup watchdog (the client dedupes by index, so
    the two threads must not reuse an index)."""
    with job["ilock"]:
        idx = (job["lines"][-1][0] + 1) if job["lines"] else 1
        job["lines"].append((idx, line))
    _broadcast(job, {"type": "log", "i": idx, "line": line})
    return idx


def _reader_thread(job, proc):
    for raw in iter(proc.stdout.readline, ""):
        line = raw.rstrip("\n")
        if not line:
            continue
        job["last_out"] = time.time()
        ph = _classify_phase(line)
        if ph and ph != job["phase"]:
            job["phase"] = ph
            _broadcast(job, {"type": "phase", "phase": ph})
        _append_line(job, line)
    try:
        proc.stdout.close()
    except Exception:
        pass
    rc = proc.wait()
    job["returncode"] = rc
    job["ended"] = time.time()
    if job["status"] != "canceled":
        job["status"] = "done" if rc == 0 else "failed"
    # rc < 0 means killed by signal (-9 = SIGKILL); surface it in the console so a
    # silent death isn't mistaken for "nothing happened".
    print(f"[viz_server] export job {job['id']} ended: status={job['status']} rc={rc}")
    if rc and rc < 0:
        _append_line(job, f"[server] process was killed by signal {-rc}")
    if rc == 0:
        try:
            _refresh_runs()  # newly-generated run must show up in the browser now
        except Exception:
            pass
    _broadcast(job, {"type": "status", "status": job["status"], "returncode": rc})
    _broadcast(job, None)  # sentinel: terminate open streams


def start_export(opts):
    global CURRENT_JOB
    with _job_lock:
        # Busy only if the previous job's process is genuinely still alive. If it
        # died hard (e.g. OOM-killed) before the reader updated status, poll() is
        # not None — reap it so a new generation isn't wedged behind a dead job.
        if CURRENT_JOB and CURRENT_JOB["status"] == "running":
            if CURRENT_JOB["proc"].poll() is None:
                return None, CURRENT_JOB["id"]
            CURRENT_JOB["status"] = "failed"
            print(f"[viz_server] reaped dead export job {CURRENT_JOB['id']} "
                  f"(rc={CURRENT_JOB['proc'].returncode})")
        run_dir = opts["run_dir"]
        dataset = os.path.basename(run_dir)
        embeddings = opts.get("embeddings", "both")
        model = _resolve_model(run_dir, embeddings)
        cmd = [sys.executable, "-u", "-m", "pidsmaker.vizgen.web.export",
               model, dataset, "--run", run_dir, "--embeddings", embeddings]
        if opts.get("all_epochs"):
            cmd.append("--all_epochs")
        if opts.get("method"):
            cmd += ["--method", str(opts["method"])]
        if opts.get("max_benign"):
            cmd += ["--max_benign", str(opts["max_benign"])]
        if opts.get("max_attack"):
            cmd += ["--max_attack", str(opts["max_attack"])]

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        print(f"[viz_server] starting export: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=PIDSMAKER_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            start_new_session=True,   # own process group, so cancel kills children too
        )
        job = {
            "id": uuid.uuid4().hex[:12],
            "run_dir": run_dir, "dataset": dataset, "model": model,
            "cmd": " ".join(cmd), "status": "running", "phase": "start",
            "started": time.time(), "ended": None, "returncode": None,
            "last_out": time.time(), "ilock": threading.Lock(),
            "lines": deque(maxlen=5000), "subscribers": [], "proc": proc,
        }
        job["lines"].append((0, "[server] launching generation…"))
        CURRENT_JOB = job
        threading.Thread(target=_reader_thread, args=(job, proc), daemon=True).start()
        threading.Thread(target=_watchdog_thread, args=(job,), daemon=True).start()
        return job, None


def _watchdog_thread(job):
    """Abort a job wedged in startup so it can't block the next generation forever.
    Phase still "start" means it never reached any real work; if it's also been
    silent past STARTUP_STALL_LIMIT the process is hung (e.g. a busy GPU during CUDA
    init) rather than merely slow. Only the "start" phase is guarded — real work may
    legitimately be quiet for a while."""
    while job["status"] == "running":
        time.sleep(5)
        if job["status"] != "running":
            return
        if job["phase"] == "start" and time.time() - job["last_out"] >= STARTUP_STALL_LIMIT:
            _append_line(job, f"[server] aborting — no progress for {STARTUP_STALL_LIMIT}s during "
                              f"startup; the GPU may be busy (check nvidia-smi). Try again.")
            job["status"] = "failed"
            try:
                os.killpg(os.getpgid(job["proc"].pid), signal.SIGKILL)
            except Exception:
                try:
                    job["proc"].kill()
                except Exception:
                    pass
            return


def cancel_export():
    job = CURRENT_JOB
    if job and job["status"] == "running":
        job["status"] = "canceled"
        proc = job["proc"]

        def _kill():   # escalate SIGTERM -> SIGKILL so a canceled job can't linger
            for sig, wait in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
                if proc.poll() is not None:
                    return
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except Exception:
                    try:
                        proc.terminate() if sig == signal.SIGTERM else proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=wait)
                except Exception:
                    pass
        threading.Thread(target=_kill, daemon=True).start()
        return True
    return False


def job_snapshot(job, tail=400):
    return {
        "id": job["id"], "status": job["status"], "phase": job["phase"],
        "model": job["model"], "dataset": job["dataset"], "run_dir": job["run_dir"],
        "cmd": job["cmd"], "started": job["started"],
        "elapsed": round((job["ended"] or time.time()) - job["started"], 1),
        "returncode": job["returncode"],
        "tail": [{"i": i, "line": l} for i, l in list(job["lines"])[-tail:]],
        "phases": PHASE_KEYS, "phase_labels": PHASE_LABELS,
    }


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


# --------------------------------------------------------------------------- #
# Score distribution — rendered server-side with the exact native matplotlib
# figure (so it is pixel-identical to the desktop viewer), returned as PNG.
# --------------------------------------------------------------------------- #

_SCOREDIST_CACHE = {}


def render_score_distribution(points_path):
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    get_info(points_path)  # ensure cache
    _, meta_p, info_p = _cache_paths(points_path)
    # the plot is deterministic per run — cache the rendered PNG so re-opening the
    # dialog is instant instead of re-rendering (matplotlib over millions of scores).
    store_npz = meta_p + ".store.npz"
    ck = (points_path, os.path.getmtime(store_npz) if os.path.exists(store_npz) else 0)
    if ck in _SCOREDIST_CACHE:
        return _SCOREDIST_CACHE[ck]
    store = load_meta(points_path)
    with open(info_p) as f:
        info = json.load(f)

    raw_scores = np.asarray(store["scores"], dtype=float)
    labels = np.asarray(store["labels"], dtype=int)
    type_enum = np.asarray(store["type_enum"], dtype=int)  # 0 proc,1 file,2 netflow,3 other
    campaigns = store.get("campaigns")
    num_campaigns = info.get("num_campaigns", 0)
    cur_thr = (info.get("detection_cost") or {}).get("current_threshold")

    raw_min, raw_max = float(np.min(raw_scores)), float(np.max(raw_scores))
    span = (raw_max - raw_min) if raw_max > raw_min else 1.0
    norm_scores = (raw_scores - raw_min) / span

    benign_type_colors = {"subject": "#1b5e20", "file": "#bbbbbb", "netflow": "#a5d6a7"}
    alpha_val = 0.7
    attack_colors = {0: "black", 1: "red", 2: "#377eb8"}

    # Split benign scores by node type with vectorised numpy masks instead of a
    # Python loop over every node — the loop was O(N) in the interpreter (~27s at
    # 10M). Only the (few) attack nodes still need a per-node loop for campaign id.
    benign = labels == 0
    benign_by_type = {
        "subject": norm_scores[benign & (type_enum == 0)],
        "netflow": norm_scores[benign & (type_enum == 2)],
        "file": norm_scores[benign & ((type_enum == 1) | (type_enum == 3))],
    }
    attack_scores = {}
    for i in np.nonzero(labels != 0)[0]:
        cids = campaigns[i] if campaigns else [0]
        raw_at = cids[0] if cids else 0
        try:
            attack_type = int(raw_at)
        except (ValueError, TypeError):
            attack_type = raw_at
        attack_scores.setdefault(attack_type, []).append(norm_scores[i])

    bins = np.linspace(0, 1, 75)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    legend_patches = []
    for ntype in ("subject", "file", "netflow"):
        vals = benign_by_type.get(ntype, [])
        if len(vals) == 0:
            continue
        color = benign_type_colors[ntype]
        label = f"Benign ({ntype})"
        ax.hist(vals, bins=bins, alpha=alpha_val, label=label, color=color,
                edgecolor="black", linewidth=0.5, log=True)
        legend_patches.append(Patch(facecolor=color, edgecolor="black", alpha=alpha_val, label=label))

    def _atk_label(at):
        return f"Attack #{at + 1}" if isinstance(at, int) else f"Attack {at}"

    for attack_type, values in attack_scores.items():
        ax.hist(values, bins=bins, alpha=alpha_val, label=_atk_label(attack_type),
                color=attack_colors.get(attack_type, "black"), edgecolor="black",
                linewidth=0.5, log=True)
    for atype in sorted(attack_scores.keys(), key=lambda x: (isinstance(x, str), x)):
        legend_patches.append(Patch(facecolor=attack_colors.get(atype, "black"), edgecolor="black",
                                    alpha=alpha_val, label=_atk_label(atype)))

    # Good Zone: precision >= 50% AND all campaigns detected (200-point sweep)
    n_curve_points = 200
    precision_cut = 0.5
    thresholds_norm = np.linspace(0, 1, n_curve_points)
    thresholds_raw = raw_min + thresholds_norm * span
    precision_curve = np.zeros(n_curve_points)
    det_curve = np.zeros(n_curve_points)

    mal_scores_sorted = np.sort(raw_scores[labels == 1])
    ben_scores_sorted = np.sort(raw_scores[labels == 0])
    mal_campaign_ids = [campaigns[i] if campaigns else [] for i in np.where(labels == 1)[0]]
    mal_order = np.argsort(raw_scores[labels == 1])
    mal_campaigns_sorted = [mal_campaign_ids[i] for i in mal_order]

    for ti, thr in enumerate(thresholds_raw):
        tp = len(mal_scores_sorted) - np.searchsorted(mal_scores_sorted, thr, side="left")
        fp = len(ben_scores_sorted) - np.searchsorted(ben_scores_sorted, thr, side="left")
        precision_curve[ti] = tp / (tp + fp + 1e-12)
        above_idx = np.searchsorted(mal_scores_sorted, thr, side="left")
        detected_camps = set()
        for cids in mal_campaigns_sorted[above_idx:]:
            detected_camps.update(cids)
        det_curve[ti] = len(detected_camps) / max(num_campaigns, 1)

    mask = (precision_curve >= precision_cut) & (det_curve >= 1.0 - 1e-12)
    if np.any(mask):
        idx = np.where(mask)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        shaded = False
        for run in runs:
            lbl = "Good Zone (P≥50% & All Attacks)" if not shaded else None
            ax.axvspan(thresholds_norm[run[0]], thresholds_norm[run[-1]], color="gray", alpha=0.2, label=lbl)
            shaded = True

    if cur_thr is not None:
        norm_thresh = (cur_thr - raw_min) / span
        if 0.0 <= norm_thresh <= 1.0:
            ax.axvline(x=norm_thresh, color="black", linestyle="--", linewidth=1.5,
                       label=f"Threshold: {norm_thresh:.2f}")

    ax.set_xlabel("Node anomaly scores", fontsize=12)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=12)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, frameon=True, fancybox=True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    png = buf.read()
    if len(_SCOREDIST_CACHE) > 8:
        _SCOREDIST_CACHE.clear()
    _SCOREDIST_CACHE[ck] = png
    return png


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(STATIC_DIR, fn)


@app.route("/favicon.ico")
def favicon():
    return Response(b"", mimetype="image/x-icon")


# Run-browser discovery scans the whole artifacts tree (~2 s), and the browser
# hits it on every page load / Run Browser open. Cache it and serve instantly,
# refreshing in the background when stale (stale-while-revalidate). A finished
# generation refreshes it synchronously so the new run shows up immediately.
_RUNS_CACHE = {"ts": 0.0, "data": None, "refreshing": False}
_RUNS_TTL = 30.0
_runs_lock = threading.Lock()


def _refresh_runs():
    data = discover_runs()
    with _runs_lock:
        _RUNS_CACHE["data"] = data
        _RUNS_CACHE["ts"] = time.time()
        _RUNS_CACHE["refreshing"] = False
    return data


def get_runs_cached(force=False):
    with _runs_lock:
        data = _RUNS_CACHE["data"]
        age = time.time() - _RUNS_CACHE["ts"]
        refreshing = _RUNS_CACHE["refreshing"]
    if force or data is None:
        return _refresh_runs()  # first call / explicit refresh: scan synchronously
    if age >= _RUNS_TTL and not refreshing:
        with _runs_lock:
            _RUNS_CACHE["refreshing"] = True
        threading.Thread(target=_refresh_runs, daemon=True).start()
    return data  # serve cached (possibly slightly stale) instantly


@app.route("/api/runs")
def api_runs():
    force = request.args.get("refresh") in ("1", "true")
    return jsonify({"artifacts_root": ARTIFACTS_ROOT, "runs": get_runs_cached(force)})


@app.route("/api/run")
def api_run():
    points = safe_path(request.args.get("file"))
    viz_dir = os.path.dirname(points)
    basename = os.path.basename(points)
    manifest = load_manifest(viz_dir)

    eval_dir = os.path.dirname(viz_dir)
    cfg_path = os.path.join(eval_dir, "run_config.yml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(eval_dir), "run_config.yml")
    featurization = _featurization_method(cfg_path)

    # embeddings available in this viz dir
    files = sorted(glob.glob(os.path.join(viz_dir, "*_points.json")))
    best_epoch = None
    if manifest and manifest.get("epochs"):
        best_epoch = sorted(
            manifest["epochs"], key=lambda e: e.get("adp", 0), reverse=True
        )[0].get("epoch")

    # per-epoch metrics from the manifest, keyed by epoch string
    ep_metrics = {}
    if manifest and manifest.get("epochs"):
        for e in manifest["epochs"]:
            ep_metrics[str(e.get("epoch"))] = (e.get("adp", 0.0), e.get("disc_score", 0.0))
    best_ep_int = None
    try:
        best_ep_int = int(best_epoch) if best_epoch is not None else None
    except (TypeError, ValueError):
        best_ep_int = best_epoch

    embeddings, epochs = [], []
    w2v_path = enc_path = None
    for fp in files:
        b = os.path.basename(fp)
        entry = {"file": fp, "label": _label_for_file(b, best_epoch, featurization), "basename": b}
        embeddings.append(entry)
        if "word2vec" in b:
            w2v_path = fp
        if "encoder" in b and "epoch" not in b:
            # the bare "encoder" file == the best epoch; surface it in the selector
            if enc_path is None:
                enc_path = fp
            ad, di = ep_metrics.get(str(best_epoch), (0.0, 0.0))
            epochs.append({"epoch": best_ep_int, "file": fp, "adp": ad, "disc": di, "is_best": True})
        ep = _epoch_of(b)
        if ep is not None:
            ad, di = ep_metrics.get(str(ep), (0.0, 0.0))
            epochs.append({"epoch": ep, "file": fp, "adp": ad, "disc": di,
                           "is_best": (str(ep) == str(best_epoch))})
    epochs.sort(key=lambda e: (e["epoch"] is None, e["epoch"]))
    # pick a representative encoder for hot-swap if no bare encoder file
    if enc_path is None and epochs:
        if best_epoch is not None:
            match = [e for e in epochs if str(e["epoch"]) == str(best_epoch)]
            enc_path = (match or epochs)[-1]["file"]
        else:
            enc_path = epochs[-1]["file"]

    info = get_info(points)
    adp, disc, _ = metrics_for_file(basename, manifest)
    dataset = os.path.basename(os.path.dirname(viz_dir))
    adj = points.replace("_points.json", "_adj.json")

    # Warm the adjacency index in the background so the first "See Edges" /
    # causal / attack action after opening the run is instant instead of paying
    # a one-time parse. The build is single-flight, so a duplicate warm is cheap.
    if os.path.exists(adj) and _ADJ_INDEX.get("key", (None,))[0] != adj:
        threading.Thread(target=warm_run, args=(points,), daemon=True).start()

    return jsonify({
        "file": points,
        "dataset": dataset,
        "model": _label_for_file(basename, best_epoch, featurization),
        "featurization": featurization,
        "n": info["n"], "hops": info["hops"], "max_tw": info["max_tw"],
        "byte_offsets": info["byte_offsets"],
        "stats": dict(info["stats"], adp=adp, disc_score=disc,
                      num_campaigns=info.get("num_campaigns", 0)),
        "detection_cost": info.get("detection_cost"),
        "run_config": load_run_config_text(viz_dir),
        "embeddings": embeddings, "epochs": epochs,
        "word2vec_file": w2v_path, "encoder_file": enc_path,
        "current_epoch": (best_ep_int if (_epoch_of(basename) is None
                          and "encoder" in basename and "epoch" not in basename)
                          else _epoch_of(basename)),
        "best_epoch": best_epoch,
        "has_adj": os.path.exists(adj), "adj_file": adj if os.path.exists(adj) else None,
    })


@app.route("/api/buffer")
def api_buffer():
    points = safe_path(request.args.get("file"))
    get_info(points)  # ensure cache built
    bin_p, _, _ = _cache_paths(points)
    # stream from disk (these are tens-to-hundreds of MB) instead of buffering
    return send_from_directory(*os.path.split(bin_p), mimetype="application/octet-stream")


@app.route("/api/neighbors")
def api_neighbors():
    """Edges incident to a single node — for the 'See Edges' dialog.

    Returns just this node's adjacency slice instead of the whole graph, so the
    browser no longer downloads the entire (tens-to-hundreds of MB) adj file.
    """
    points = safe_path(request.args.get("file"))
    try:
        node = int(request.args.get("node"))
    except (TypeError, ValueError):
        abort(400, "bad node id")
    index = get_adj_index(points)
    if index is None:
        abort(404, "no adjacency data for this run; generate viz data first")
    store = load_meta(points)
    edges = index.neighbors(node)
    # Enrich each edge with its neighbour's metadata (the client only holds a
    # prefix). A hub node can have thousands of edges, so resolve every neighbour
    # id → store row in ONE vectorised searchsorted rather than one per edge; only
    # the per-row string decode (unavoidable) stays in the loop.
    if edges and len(store.uniq_ids):
        nb = np.fromiter((e["nb"] for e in edges), dtype=np.int64, count=len(edges))
        pos = np.clip(np.searchsorted(store.uniq_ids, nb), 0, len(store.uniq_ids) - 1)
        match = store.uniq_ids[pos] == nb
        first = store.first_idx[pos]
        # A hub node can have tens of thousands of edge instances but only a few
        # hundred distinct neighbours, all carrying the same metadata — decode each
        # neighbour's row ONCE and reuse it, instead of re-slicing the same strings
        # per edge (the dominant cost when "See Edges" is slow on a hub node).
        row_cache = {}
        for e, ok, fi in zip(edges, match, first):
            if ok:
                fi = int(fi)
                r = row_cache.get(fi)
                if r is None:
                    r = row_cache[fi] = store.row(fi)
                e.update(r)
            else:
                e.update(_OFF_GRAPH)
    elif edges:
        for e in edges:
            e.update(_OFF_GRAPH)
    return jsonify({"center": node, "edges": edges})


@app.route("/api/node")
def api_node():
    """Full metadata for a small set of nodes, fetched on demand when a node is
    selected. Keyed by buffer ROW index (`idxs`), so the client needn't hold the
    node-id column up front. Each lookup is O(1), so selecting a node is instant
    even at 20M."""
    points = safe_path(request.args.get("file"))
    store = load_meta(points)
    rows = {}
    for s in (request.args.get("idxs") or "").split(","):
        s = s.strip()
        try:
            r = store.node_at(int(s)) if s else None
        except ValueError:
            r = None
        if r is not None:
            rows[s] = r
    return jsonify({"rows": rows})


@app.route("/api/search")
def api_search():
    """Buffer rows matching a single query (id or path/cmd substring), server-side
    so the client never downloads the metadata to search it. Rows == buffer
    indices, so the client filters the cloud by index."""
    points = safe_path(request.args.get("file"))
    q = request.args.get("q") or ""
    try:
        limit = min(int(request.args.get("limit", 200000)), 500000)
    except (TypeError, ValueError):
        limit = 200000
    store = load_meta(points)
    rows = store.search([q], limit)
    return jsonify({"rows": rows, "capped": len(rows) >= limit})


@app.route("/api/filter", methods=["POST"])
def api_filter():
    """Buffer rows matching ANY of many terms (the CSV filter) — same server-side
    blob scan as /api/search, batched."""
    body = request.get_json(force=True, silent=True) or {}
    points = safe_path(body.get("file"))
    terms = body.get("terms") or []
    if not isinstance(terms, list):
        terms = []
    store = load_meta(points)
    rows = store.search(terms, 500000)
    return jsonify({"rows": rows, "capped": len(rows) >= 500000})


@app.route("/api/causal")
def api_causal():
    """Server-side causal subgraph from a node (time-respecting traversal).

    Returns the reached node ids for the chronological table, plus a small
    induced subgraph (origin + malicious nodes, capped) for the directed-graph
    view — so the client renders both without holding the full adjacency.
    """
    points = safe_path(request.args.get("file"))
    try:
        start = int(request.args.get("node"))
    except (TypeError, ValueError):
        abort(400, "bad node id")
    index = get_adj_index(points)
    if index is None:
        abort(404, "no adjacency data for this run; generate viz data first")

    store = load_meta(points)
    start_tw = store.first_tw_of(start)
    ids = causal_trace(index, start, start_tw)

    # rows enriched from full metadata (the table works even for nodes outside
    # the client's loaded prefix)
    rows = [{"id": nid, **store.row(store.first_index(nid))} for nid in ids]

    # graph view: origin + malicious nodes only, capped for readability
    graph_nodes = [n for n in ids if n == start or store.label_of(n) != 0][:200]
    nset = set(graph_nodes)
    graph_edges = []
    for u in graph_nodes:
        for e in index.neighbors(u):
            if e["dir"] == "out" and e["nb"] in nset:
                graph_edges.append([u, e["nb"]])
    return jsonify({
        "count": len(ids), "rows": rows,
        "graph_nodes": graph_nodes, "graph_edges": graph_edges,
    })


@app.route("/api/attack_pairs")
def api_attack_pairs():
    """Unique malicious<->malicious edges (as [u, v] id pairs) for the 3D attack
    overlay. Small (bounded by the attack subgraph); computed from the cached
    adjacency + labels so the client needn't load the whole graph."""
    points = safe_path(request.args.get("file"))
    index = get_adj_index(points)
    if index is None:
        abort(404, "no adjacency data for this run; generate viz data first")
    mal = load_meta(points).malicious_ids()
    pairs, seen = [], set()
    for u in mal:
        for e in index.neighbors(u):
            v = e["nb"]
            if v not in mal:
                continue
            k = (u, v) if u < v else (v, u)
            if k in seen:
                continue
            seen.add(k)
            pairs.append([u, v])
    return jsonify({"pairs": pairs})


# ----- viz generation (export subprocess) ----- #

# --------------------------------------------------------------------------- #
# Campaign attack graph — built from the run's own viz data: the adjacency
# (edges, with relation type `et` and direction `dir`) and the ground-truth
# node labels and paths. No edge_scores / indexid2msg needed, so it works for
# any run that has viz data. Hop 0 = malicious<->malicious edges; hops 1/2/3
# add neighbour context via edge sampling (300/200/100). DAG-enforced.
# --------------------------------------------------------------------------- #

_CAMPAIGN_CACHE = {}
HOP_EDGE_COLOR = ["#ff4d4d", "#ffb3b3", "#cccccc", "#666666"]


def _campaign_node_color(ntype, is_mal):
    if not is_mal:
        return "#808080"
    if "process" in ntype:
        return "#ff4d4d"
    if "netflow" in ntype or "network" in ntype or "socket" in ntype:
        return "#4da6ff"
    if "file" in ntype:
        return "#4dff4d"
    return "#ffcc00"


def build_campaign_graph(points_path):
    """Cached, single-flight campaign graph. The lock stops a background warm and
    a user click from building it twice at once (which doubled the wall time)."""
    index = get_adj_index(points_path)
    if index is None:
        return None
    adj_path, _, _ = _adj_index_paths(points_path)
    _, meta_p, _ = _cache_paths(points_path)
    key = (adj_path, os.path.getmtime(adj_path), os.path.getmtime(meta_p))
    if _CAMPAIGN_CACHE.get("key") == key:
        return _CAMPAIGN_CACHE["data"]
    with _build_lock("campaign:" + adj_path):
        if _CAMPAIGN_CACHE.get("key") == key:  # built while we waited
            return _CAMPAIGN_CACHE["data"]
        data = _compute_campaign(points_path, index)
        _CAMPAIGN_CACHE["key"] = key
        _CAMPAIGN_CACHE["data"] = data
        return data


def _compute_campaign(points_path, index):
    import random

    store = load_meta(points_path)
    types, paths = store["types"], store["paths"]
    malicious = store.malicious_ids()
    if not malicious:
        return {"nodes": [], "links": []}

    def info(nid):
        i = store.first_index(nid)
        if i < 0:
            return "node", f"node: {nid}"
        t = (types[i] or "node").lower()
        name = paths[i] or str(nid)
        if "netflow" not in t:
            name = name.split("/")[-1] or name
        if len(name) > 40:
            name = "..." + name[-37:]
        return t, f"{t}: {name}"

    def directed(u, e):
        return (e["nb"], u) if e.get("dir") == "in" else (u, e["nb"])

    random.seed(42)

    # hop 0: malicious <-> malicious edges
    hop = {}
    hop0_edges, eseen = [], set()
    for u in malicious:
        for e in index.neighbors(u):
            if e["nb"] not in malicious:
                continue
            s, t = directed(u, e)
            k = (s, t, e.get("et", ""))
            if k in eseen:
                continue
            eseen.add(k)
            hop0_edges.append((s, t, e.get("et", "")))
            hop[s] = 0
            hop[t] = 0
    # Cap hop-0 like the other hops. add_edge() runs an O(V+E) nx.find_cycle per
    # edge, so an uncapped hop-0 (many interconnected malicious nodes) makes the
    # whole build O(E²) and can wedge for minutes. 500 edges keeps it bounded and
    # is plenty for a readable graph.
    HOP0_CAP = 500
    if len(hop0_edges) > HOP0_CAP:
        hop0_edges = random.sample(hop0_edges, HOP0_CAP)
        hop = {}
        for s, t, _et in hop0_edges:
            hop[s] = 0
            hop[t] = 0
    if not hop:
        for nid in malicious:
            hop[nid] = 0

    def boundary(inside, exclude, cap):
        out, s2 = [], set()
        for u in inside:
            for e in index.neighbors(u):
                if e["nb"] in exclude:
                    continue
                s, t = directed(u, e)
                k = (s, t, e.get("et", ""))
                if k in s2:
                    continue
                s2.add(k)
                out.append((s, t, e.get("et", "")))
        return random.sample(out, cap) if len(out) > cap else out

    seen = set(hop)
    frontier = set(hop)
    hop_edges = {0: hop0_edges}
    for h, cap in ((1, 300), (2, 200), (3, 100)):
        edges = boundary(frontier, seen, cap)
        hop_edges[h] = edges
        new = set()
        for s, t, _et in edges:
            for nid in (s, t):
                if nid not in seen:
                    hop[nid] = h
                    new.add(nid)
        seen |= new
        frontier = new
        if not new:
            break

    nodes, links = {}, []
    seen_links = set()

    def add_node(nid, h):
        if nid in nodes:
            return
        t, lbl = info(nid)
        nodes[nid] = {"id": str(nid), "label": lbl, "hop": h, "color": _campaign_node_color(t, h == 0)}

    def add_edge(s, t, et, h):
        # Orient each edge forward in (hop, id) order instead of dropping backward
        # ones: a strict endpoint order keeps the graph acyclic without discarding
        # edges, so a connected subgraph stays connected.
        ks, kt = (hop.get(s, 1 << 30), s), (hop.get(t, 1 << 30), t)
        if ks == kt:
            return
        if ks > kt:
            s, t = t, s
        if (s, t) in seen_links:
            return
        seen_links.add((s, t))
        links.append({"source": str(s), "target": str(t), "label": et, "hop": h, "color": HOP_EDGE_COLOR[h]})

    for h in (0, 1, 2, 3):
        for nid, hh in list(hop.items()):
            if hh == h:
                add_node(nid, h)
        for s, t, et in hop_edges.get(h, []):
            add_edge(s, t, et, h)

    return {"nodes": list(nodes.values()), "links": links}


@app.route("/api/campaign")
def api_campaign():
    points = safe_path(request.args.get("file"))
    try:
        data = build_campaign_graph(points)
    except Exception as e:
        abort(500, f"campaign graph build failed: {type(e).__name__}: {e}")
    if data is None:
        abort(404, "no adjacency data for this run; generate viz data first")
    return jsonify(data)


@app.route("/api/scoredist")
def api_scoredist():
    points = safe_path(request.args.get("file"))
    png = render_score_distribution(points)
    return Response(png, mimetype="image/png")


@app.route("/api/export", methods=["POST"])
def api_export():
    if not ALLOW_EXPORT:
        abort(403, "viz generation disabled (set PIDS_VIZ_ALLOW_EXPORT=1)")
    body = request.get_json(force=True, silent=True) or {}
    run_dir = safe_path(body.get("run_dir"), kind="dir")
    opts = {
        "run_dir": run_dir,
        "embeddings": body.get("embeddings", "both"),
        "all_epochs": bool(body.get("all_epochs")),
        "method": body.get("method") or "umap",
        "max_benign": body.get("max_benign"),
        "max_attack": body.get("max_attack"),
    }
    job, busy_id = start_export(opts)
    if job is None:
        return jsonify({"error": "an export is already running", "job_id": busy_id}), 409
    return jsonify({"job_id": job["id"]})


@app.route("/api/export/status")
def api_export_status():
    if CURRENT_JOB is None:
        return jsonify({"job": None, "allowed": ALLOW_EXPORT})
    return jsonify({"job": job_snapshot(CURRENT_JOB), "allowed": ALLOW_EXPORT})


@app.route("/api/export/cancel", methods=["POST"])
def api_export_cancel():
    return jsonify({"canceled": cancel_export()})


@app.route("/api/export/stream")
def api_export_stream():
    job = CURRENT_JOB
    if job is None:
        abort(404, "no export job")
    q = queue.Queue()
    job["subscribers"].append(q)

    @stream_with_context
    def gen():
        yield _sse({"type": "snapshot", **job_snapshot(job)})
        if job["status"] != "running":
            try:
                job["subscribers"].remove(q)
            except ValueError:
                pass
            return
        try:
            while True:
                # Heartbeat: if no log arrives for 15s (e.g. a big torch.load or a
                # long silent edge scan), emit an SSE comment so the tunnel/browser
                # keeps the stream open instead of dropping it as idle.
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if msg is None:
                    break
                yield _sse(msg)
        finally:
            try:
                job["subscribers"].remove(q)
            except ValueError:
                pass

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    ap = argparse.ArgumentParser(description="PIDSMaker web visualizer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[viz_server] artifacts root: {ARTIFACTS_ROOT}")
    print(f"[viz_server] open http://{args.host}:{args.port}/")
    # prime the run-browser cache so the first page load is instant
    threading.Thread(target=lambda: get_runs_cached(force=True), daemon=True).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
