#!/usr/bin/env python3
"""CLI entry point for interactive 3D embedding visualization.

Usage:
    python scripts/embedding_viz.py <model> <dataset> [options]

Examples:
    # Word2Vec raw embeddings (default)
    python scripts/embedding_viz.py orthrus CADETS_E3

    # GNN encoder embeddings
    python scripts/embedding_viz.py orthrus CADETS_E3 --embeddings encoder

    # Both views
    python scripts/embedding_viz.py orthrus CADETS_E3 --embeddings both

    # Custom sampling
    python scripts/embedding_viz.py orthrus CADETS_E3 --max_benign 10000 --max_attack all

    # t-SNE instead of UMAP
    python scripts/embedding_viz.py orthrus CADETS_E3 --method tsne
"""

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml
from yacs.config import CfgNode as CN

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.utils.utils import get_device, get_node_to_path_and_type, log
from pidsmaker.vizgen.dimensionality_reduction import reduce_to_3d
from pidsmaker.vizgen.embed_exporter import (
    extract_encoder_embeddings,
    extract_word2vec_embeddings,
    smart_sample,
)
from pidsmaker.vizgen.html_builder import build_html


def load_viz_config():
    """Load viz_config.yml defaults."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "viz_config.yml"
    )
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("embedding_viz", {})
    return {}


def _get_artifacts_root():
    """Return the correct artifacts root for Docker vs Host."""
    if os.path.exists("/home/artifacts"):
        return "/home/artifacts"
    pidsmaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.environ.get("PIDS_ARTIFACTS_DIR", os.path.join(pidsmaker_root, "artifacts"))


def _find_manifests(dataset):
    """Locate all viz_manifest.json files for a given dataset."""
    artifacts_root = _get_artifacts_root()
    manifests = []
    for base in ("evaluation/evaluation", "detection/evaluation"):
        pattern = os.path.join(artifacts_root, base, "*", dataset, "viz_manifest.json")
        manifests.extend(glob.glob(pattern))
    return manifests


def find_models(dataset="CADETS_E3", run_dir=None):
    """Discover evaluated model epochs for visualization.

    Reads viz_manifest.json (written by set_task_to_done) for exact artifact
    paths.  Falls back to glob-based discovery if no manifest exists yet.

    If *run_dir* is given (a specific run's eval task dir or its
    viz_manifest.json), that run is used directly instead of the latest one
    found for the dataset — lets the user export a chosen run from artifacts.
    """
    if run_dir:
        mp = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "viz_manifest.json")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"viz_manifest.json not found for --run {run_dir}")
        print(f"  Using run manifest: {mp}")
        with open(mp, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return _models_from_manifest(manifest)

    print(f"Scanning for best models for dataset: {dataset}...")
    manifests = _find_manifests(dataset)

    if manifests:
        # Pick the most recently modified manifest
        manifests.sort(key=os.path.getmtime, reverse=True)
        manifest_path = manifests[0]
        print(f"  Using manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return _models_from_manifest(manifest)

    # --- Fallback: glob-based discovery (backward compat) ---
    print("  WARNING: No viz_manifest.json found — falling back to glob discovery.")
    print("  WARNING: Encoder embeddings may fail (no trained_models_dir or preprocessed_graphs_dir available).")
    return _models_from_glob(dataset)


def _models_from_manifest(manifest):
    """Build a models list from a viz_manifest.json."""
    models = []
    for entry in manifest.get("epochs", []):
        stats_path = entry["stats_path"]
        if not os.path.exists(stats_path):
            continue
        try:
            stats = torch.load(stats_path, map_location="cpu")
        except Exception:
            continue

        adp = stats.get("adp_score", 0)
        epoch_str = entry["epoch"]

        model_path = None
        tm_dir = manifest.get("trained_models_dir")
        if tm_dir and os.path.exists(tm_dir):
            mp = os.path.join(tm_dir, f"model_epoch_{epoch_str}")
            if os.path.exists(mp):
                model_path = mp

        models.append({
            "name": f"Epoch_{epoch_str}_ADP_{adp:.3f}",
            "path": model_path,
            "adp": adp,
            "epoch": epoch_str,
            "stats_path": stats_path,
            "scores_path": entry.get("scores_path"),
            "edge_scores_path": entry.get("scores_path"),  # compat alias
            "edge_losses_dir": manifest.get("edge_losses_dir"),
            "trained_models_dir": tm_dir,
            "preprocessed_graphs_dir": manifest.get("preprocessed_graphs_dir"),
            "eval_task_path": manifest.get("eval_task_path"),
        })

    models.sort(key=lambda x: x["adp"], reverse=True)
    return models[:6]


def _models_from_glob(dataset):
    """Legacy glob-based model discovery."""
    artifacts_root = _get_artifacts_root()
    eval_patterns = [
        os.path.join(artifacts_root, "detection/evaluation/*", dataset,
                      "precision_recall_dir/stats_model_epoch_*.*"),
        os.path.join(artifacts_root, "evaluation/evaluation/*", dataset,
                      "precision_recall_dir/stats_model_epoch_*.*"),
    ]
    models_found = []
    for pattern in eval_patterns:
        for f in glob.glob(pattern):
            if f.endswith(".png"):
                continue
            try:
                s = torch.load(f, map_location="cpu")
                adp = s.get("adp_score", 0)
                epoch_str = os.path.basename(f).replace("stats_model_epoch_", "").rsplit(".", 1)[0]
                models_found.append({
                    "name": f"Epoch_{epoch_str}_ADP_{adp:.3f}",
                    "path": None,
                    "adp": adp,
                    "epoch": epoch_str,
                    "stats_path": f,
                    "edge_scores_path": f.replace("stats_", "scores_"),
                })
            except Exception:
                continue

    models_found.sort(key=lambda x: x["adp"], reverse=True)
    return models_found[:6]

def get_malicious_node_ids(cfg):
    """Get ground-truth malicious node IDs."""
    from pidsmaker.utils.labelling import get_ground_truth

    gt_nids, _, _ = get_ground_truth(cfg)
    log(f"Ground truth: {len(gt_nids)} malicious nodes")
    return gt_nids


def run_visualization(args, cfg):
    """Main visualization pipeline."""
    viz_cfg = load_viz_config()

    # CLI args override config
    embeddings = args.embeddings or viz_cfg.get("embeddings", "word2vec")
    method = args.method or viz_cfg.get("method", "umap")
    max_benign = args.max_benign or viz_cfg.get("max_benign_nodes", "all")
    max_attack = args.max_attack or viz_cfg.get("max_attack_nodes", "all")
    default_hops = int(viz_cfg.get("default_hops", 0))

    # Output directory logic — use manifest eval_task_path if available
    models = find_models(cfg.dataset.name, run_dir=getattr(args, "run", None))
    artifacts_root = _get_artifacts_root()

    eval_task_path = None
    if models and models[0].get("eval_task_path"):
        eval_task_path = models[0]["eval_task_path"]

    if eval_task_path:
        out_dir = os.path.join(eval_task_path, "viz")
    else:
        out_dir = os.path.join(artifacts_root, "viz")

    os.makedirs(out_dir, exist_ok=True)
    log(f"Saving visualization artifacts to {out_dir}")

    # Get ground truth
    malicious_ids = get_malicious_node_ids(cfg)

    modes = []
    if embeddings in ("word2vec", "both"):
        modes.append({"type": "word2vec", "suffix": "word2vec", "title": f"{cfg.dataset.name} — Word2Vec Raw Embeddings"})
    if embeddings in ("encoder", "both"):
        if not models:
            raise ValueError(f"No trained models found for dataset {cfg.dataset.name}")
        if hasattr(args, 'epoch') and args.epoch is not None:
            # Single-epoch mode: filter to the requested epoch
            models_to_run = [m for m in models if str(m.get('epoch')) == str(args.epoch)]
            if not models_to_run:
                raise ValueError(f"Epoch {args.epoch} not found in available models")
        elif args.all_epochs:
            models_to_run = models
        else:
            models_to_run = [models[0]]
        for m_info in models_to_run:
            ep = m_info.get("epoch", "latest")
            # For the default/latest model, keep the standard suffix for compatibility
            # unless --all_epochs is passed, in which case we append the epoch.
            if not args.all_epochs and m_info == models[0]:
                suffix = "encoder"
            else:
                suffix = f"encoder_epoch_{ep}"
            modes.append({
                "type": "encoder",
                "suffix": suffix,
                "title": f"{cfg.dataset.name} — GNN Encoder (Epoch {ep})",
                "model_info": m_info
            })

    cached_graph_data = None

    encoder_jobs = [j for j in modes if j["type"] == "encoder"]
    last_encoder_suffix = encoder_jobs[-1]["suffix"] if encoder_jobs else None

    for job in modes:
        mode_type = job["type"]
        log(f"{'='*60}", pre_return_line=True)
        log(f"Running {job['suffix']} embedding extraction...")
        log(f"{'='*60}")

        if mode_type == "word2vec":
            result = extract_word2vec_embeddings(cfg, malicious_ids)
            title = job["title"]
        else:
            # Load model and data (only once)
            if cached_graph_data is None:

                # Try loading from disk cache first (saved by training_loop.py)
                # This avoids recomputing the entire TGN batching pipeline from scratch
                _cache_dir = cfg.batching._preprocessed_graphs_dir
                if job.get("model_info", {}).get("preprocessed_graphs_dir"):
                    _cache_dir = job["model_info"]["preprocessed_graphs_dir"]
                _cache_file = os.path.join(_cache_dir, "torch_graphs.pkl")
                _viz_cache_file = os.path.join(_cache_dir, "viz_test_graphs.pkl")

                if os.path.exists(_viz_cache_file):
                    log(f"Loading preprocessed test graphs from viz cache: {_viz_cache_file}")
                    test_data, max_node_num = torch.load(_viz_cache_file)
                elif os.path.exists(_cache_file):
                    log(f"Loading preprocessed graphs from cache: {_cache_file}")
                    _, _, test_data, max_node_num = torch.load(_cache_file)
                else:
                    log("No cached graphs found, recomputing...")
                    try:
                        from pidsmaker.tasks.batching import get_preprocessed_graphs
                    except ImportError:
                        from pidsmaker.detection.graph_preprocessing import get_preprocessed_graphs

                    tmp_train, tmp_val, test_data, max_node_num = get_preprocessed_graphs(cfg)
                    del tmp_train
                    del tmp_val
                    gc.collect()

                gc.collect()
                cached_graph_data = (test_data, max_node_num)

            device = get_device(cfg)
            test_data, max_node_num = cached_graph_data

            from pidsmaker.factory import build_model

            m_info = job["model_info"]
            try:
                model = build_model(
                    data_sample=test_data[0][0],
                    device=device,
                    cfg=cfg,
                    max_node_num=max_node_num,
                )
                sd_path = m_info.get('path')
                if sd_path and os.path.isdir(sd_path):
                    sd_path = os.path.join(sd_path, "state_dict.pkl")
                if sd_path:
                    model.load_state_dict(
                        torch.load(sd_path, map_location=device, weights_only=False)
                    )
                    log(f"Loaded model weights from {sd_path}")
                else:
                    log(f"Warning: No valid model path found for epoch {m_info.get('epoch')}. Using uninitialized weights.")
            except Exception as e:
                log(f"ERROR: could not build/load the encoder for epoch {m_info.get('epoch')}: "
                    f"the saved checkpoint does not match the model architecture this config builds. "
                    f"This run was likely trained with a different config (and has no saved run_config.yml to "
                    f"reconstruct it). Skipping encoder export — word2vec, if requested, is unaffected. "
                    f"[{type(e).__name__}: {str(e).splitlines()[0][:160]}]")
                continue

            epoch_str = m_info.get("epoch", "0")
            detected_nodes = None
            node_anomaly_info = None
            try:
                scores_path = m_info.get("scores_path") or m_info.get("edge_scores_path")
                if scores_path and os.path.exists(scores_path):
                    from pidsmaker.vizgen.embed_exporter import parse_scores_file
                    detected_nodes, _, node_anomaly_info = parse_scores_file(scores_path, malicious_ids)
                    log(f"Evaluation Stats: Found {len(detected_nodes)} detected nodes for epoch {epoch_str}.")
            except Exception as e:
                log(f"Warning: Failed to parse evaluation stats for coloring: {e}")

            result = extract_encoder_embeddings(
                model, test_data, device, malicious_ids,
                detected_node_ids=detected_nodes,
                node_anomaly_info=node_anomaly_info
            )
            title = job["title"]

        # Sample
        result = smart_sample(result, max_benign, max_attack)

        # Dimensionality reduction (GPU-accelerated if available)
        device = get_device(cfg)
        points = reduce_to_3d(result, method=method, device=device)

        # Free massive embeddings array immediately
        result.embeddings = []
        gc.collect()

        # Node metadata
        log("Loading node metadata from database...")
        node_meta = get_node_to_path_and_type(cfg)

        # Define output path
        suffix = job["suffix"]
        out_path = os.path.join(
            out_dir,
            f"embedding_viz_{cfg.dataset.name}_{suffix}.html",
        )
        if hasattr(args, "output") and args.output:
            out_path = args.output

        # Resolve edge-type ids -> relation names (e.g. EVENT_READ) for the viewer
        from pidsmaker.utils.dataset_utils import get_rel2id

        rel = get_rel2id(cfg, from_zero=True)
        id2name = {k: v for k, v in rel.items() if isinstance(k, int)}
        typed_edges = [
            (e[0], e[1], e[2], id2name.get(e[3], "") if len(e) > 3 else "")
            for e in result.edges
        ]

        # Build HTML
        log("Building interactive HTML viewer...")
        html = build_html(
            points=points,
            edges=typed_edges,
            node_metadata=node_meta,
            title=title,
            default_hops=default_hops,
            out_path=out_path,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        log(f"Saved visualization to: {out_path}")
        log(f"File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")

        # Free memory at the end of loop iteration
        del html
        del points
        del result
        del node_meta

        # Free massive cached graphs if this was the last encoder job
        if mode_type == "encoder" and job["suffix"] == last_encoder_suffix:
            if cached_graph_data is not None:
                del cached_graph_data
                cached_graph_data = None

        gc.collect()


def _prune_unknown_keys(loaded, base, path=""):
    """Drop keys from `loaded` not present in `base`'s schema, so the rest of
    the config can still merge. Returns the dropped dotted key names."""
    dropped = []
    for key in list(loaded.keys()):
        full_key = f"{path}.{key}" if path else key
        if key not in base:
            dropped.append(full_key)
            del loaded[key]
        elif isinstance(loaded[key], dict) and isinstance(base[key], dict):
            dropped += _prune_unknown_keys(loaded[key], base[key], full_key)
    return dropped


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Embedding Visualization")
    parser.add_argument("model", type=str, help="Model config name (e.g. orthrus, velox)")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g. CADETS_E3)")
    parser.add_argument("--embeddings", type=str, choices=["word2vec", "encoder", "both"],
                        default=None, help="Embedding source (default: from viz_config.yml)")
    parser.add_argument("--method", type=str, choices=["umap", "tsne"],
                        default=None, help="DR method (default: from viz_config.yml)")
    parser.add_argument("--max_benign", type=str, default=None,
                        help="Max benign nodes or 'all'")
    parser.add_argument("--max_attack", type=str, default=None,
                        help="Max attack nodes or 'all'")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path for the HTML file")
    parser.add_argument("--all_epochs", action="store_true",
                        help="Export embeddings for all available training epochs")
    parser.add_argument("--epoch", type=str, default=None,
                        help="Export embeddings for a specific epoch only (used for memory-safe per-epoch runs)")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to a specific run's eval task dir (or its viz_manifest.json) "
                             "in the artifacts folder to export, instead of the latest for the dataset")

    args, unknown = parser.parse_known_args()

    # Build pipeline args for get_yml_cfg (positional: model dataset)
    sys.argv = [
        sys.argv[0],
        args.model,
        args.dataset,
    ] + unknown

    pipeline_args, _ = get_runtime_required_args(return_unknown_args=True)
    cfg = get_yml_cfg(pipeline_args)

    # If exporting a specific run that saved its config, reconstruct cfg from it
    # so build_model rebuilds the exact trained architecture (the repo defaults
    # may have diverged since the run was trained).
    if getattr(args, "run", None):
        run_dir = args.run[: -len("/viz_manifest.json")] if args.run.endswith(".json") else args.run
        for rc in (os.path.join(run_dir, "run_config.yml"),
                   os.path.join(os.path.dirname(run_dir), "run_config.yml")):
            if os.path.exists(rc):
                try:
                    with open(rc) as f:
                        loaded = CN.load_cfg(f)
                    dropped = _prune_unknown_keys(loaded, cfg)
                    if dropped:
                        log(f"Warning: {rc} has config key(s) no longer in the "
                            f"schema (likely trained with an older version); "
                            f"dropped and using current defaults for: "
                            f"{', '.join(dropped)}")
                    cfg.merge_from_other_cfg(loaded)
                    log(f"Reconstructed config from run's run_config.yml: {rc}")
                except Exception as e:
                    log(f"Warning: could not merge {rc} ({type(e).__name__}: {e}); "
                        f"using default '{args.model}' config.")
                break

    run_visualization(args, cfg)


if __name__ == "__main__":
    main()
