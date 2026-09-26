#!/usr/bin/env python3
"""
Evaluate a pretrained spider model on eval_set_clustering.json.

Loads the T5 student encoder from a spider checkpoint, embeds each entity,
runs leave-one-out KNN classification, and logs per-example success/failure
with predicted vs true cluster labels.

Usage:
    python scripts/eval_spider.py --model-dir /path/to/stored_models
    python scripts/eval_spider.py --model-dir /path/to/stored_models --k 10
    python scripts/eval_spider.py --model-dir /path/to/stored_models --output eval_results.json
    python scripts/eval_spider.py --model-dir /path/to/stored_models --plot
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional

# Ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

EVAL_SET_PATH = os.path.join(_PROJECT_ROOT, "config", "eval", "eval_set_clustering.json")

# Maps eval entity_type → tokenizer node type
EVAL_TYPE_MAP = {"PROC": "subject", "FILE": "file", "SOCK": "netflow"}


# ============================================================================
# MODEL LOADING & EMBEDDING
# ============================================================================

def load_model_and_tokenizer(model_dir: str, device: str):
    """Load the spider T5 student encoder and tokenizer from a checkpoint."""
    import torch
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE
    from pidsmaker.spider.models.spider import (
        ProvenanceGNNCluster,
        get_spider_encoder_config,
    )

    # Load tokenizer
    tokenizer = object.__new__(ProvenanceTokenizerBPE)
    tokenizer.expand_netflow_ips = False
    tokenizer.load(os.path.join(model_dir, "tokenizer.pt"))
    print(f"Loaded tokenizer (vocab={tokenizer.vocab_size}, max_seq_len={tokenizer.max_seq_len})")

    # Detect model size from checkpoint filename
    model_size = "mini"
    for sz in ("tiny", "mini", "med", "baseline"):
        best_path = os.path.join(model_dir, f"pretrain_{sz}_best.pt")
        regular_path = os.path.join(model_dir, f"pretrain_{sz}.pt")
        if os.path.exists(best_path) or os.path.exists(regular_path):
            model_size = sz
            break

    pt_path = os.path.join(model_dir, f"pretrain_{model_size}_best.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join(model_dir, f"pretrain_{model_size}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    # Infer hidden_dim from checkpoint
    sd = torch.load(pt_path, weights_only=True, map_location="cpu")
    # The projection MLP: projection.fc1.weight has shape [gnn_hidden_dim, d_model]
    gnn_hidden_dim = 256  # default
    for key, val in sd.items():
        if key == "projection.fc1.weight":
            gnn_hidden_dim = val.shape[0]
            break

    config = get_spider_encoder_config(tokenizer.vocab_size, model_size, tokenizer.max_seq_len)
    model = ProvenanceGNNCluster(config, gnn_hidden_dim)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded spider student ({model_size}, H={config.d_model}, gnn_H={gnn_hidden_dim}) from {pt_path}")

    return model, tokenizer, config.d_model


def compute_embeddings(
    model,
    tokenizer,
    dataset: List[dict],
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed all entities from the eval dataset. Returns [N, H] L2-normalized embeddings."""
    import torch

    node_types = [EVAL_TYPE_MAP[e["entity_type"]] for e in dataset]
    labels = [e["entity_text"] for e in dataset]

    all_token_ids = []
    for ntype, label in zip(node_types, labels):
        ids = tokenizer.tokenize_node(ntype, label)
        if not ids:
            ids = [tokenizer.pad_id]
        all_token_ids.append(ids)

    n_total = len(all_token_ids)
    all_embeddings = []

    with torch.no_grad():
        for batch_start in range(0, n_total, batch_size):
            if batch_start % (batch_size * 10) == 0:
                print(f"  Encoding {batch_start}/{n_total}...")
            batch = all_token_ids[batch_start : batch_start + batch_size]
            max_len = min(max(len(t) for t in batch), tokenizer.max_seq_len)
            B = len(batch)

            input_ids = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
            for i, tids in enumerate(batch):
                seq_len = min(len(tids), max_len)
                input_ids[i, :seq_len] = torch.tensor(tids[:seq_len])
                attention_mask[i, :seq_len] = True

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            hidden = model.modified_fwd(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=torch.full_like(input_ids, -100),
                skip_cls=True,
            )  # [B, max_len, H]

            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy())

    emb = np.concatenate(all_embeddings, axis=0)

    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb = emb / norms

    print(f"  Encoded {n_total} entities -> {emb.shape[1]}D embeddings")
    return emb


# ============================================================================
# KNN EVALUATION
# ============================================================================

def knn_classify(X: np.ndarray, labels: List[str], k: int = 5):
    """Leave-one-out KNN classification over the eval set.

    Returns:
        predictions: list of predicted cluster labels
        neighbor_info: list of dicts with KNN details per example
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    distances, indices = nn.kneighbors(X)

    predictions = []
    neighbor_info = []

    for i in range(len(X)):
        # Exclude self (index 0 in sorted neighbors)
        nbr_indices = indices[i][1 : k + 1]
        nbr_distances = distances[i][1 : k + 1]
        nbr_labels = [labels[j] for j in nbr_indices]

        vote_counts = Counter(nbr_labels)
        pred = vote_counts.most_common(1)[0][0]
        predictions.append(pred)

        neighbor_info.append(
            {
                "neighbor_labels": nbr_labels,
                "neighbor_distances": [round(float(d), 4) for d in nbr_distances],
                "vote_counts": dict(vote_counts.most_common()),
            }
        )

    return predictions, neighbor_info


def evaluate(
    dataset: List[dict],
    embeddings: np.ndarray,
    k: int = 5,
) -> dict:
    """Run KNN classification and compute metrics.

    Returns a dict with overall accuracy, per-cluster accuracy,
    and per-example results (success/failure + predicted class).
    """
    true_labels = [e["cluster"] for e in dataset]
    predictions, neighbor_info = knn_classify(embeddings, true_labels, k=k)

    # Per-example results
    examples = []
    for i, entry in enumerate(dataset):
        correct = predictions[i] == true_labels[i]
        examples.append(
            {
                "entity_type": entry["entity_type"],
                "entity_text": entry["entity_text"],
                "true_cluster": true_labels[i],
                "predicted_cluster": predictions[i],
                "correct": correct,
                "seen_during_training": entry["seen_during_training"],
                "neighbor_votes": neighbor_info[i]["vote_counts"],
                "neighbor_distances": neighbor_info[i]["neighbor_distances"],
            }
        )

    # Overall accuracy
    correct_count = sum(1 for ex in examples if ex["correct"])
    total = len(examples)
    accuracy = correct_count / total if total > 0 else 0.0

    # Per-cluster accuracy
    cluster_correct = defaultdict(int)
    cluster_total = defaultdict(int)
    for ex in examples:
        cluster_total[ex["true_cluster"]] += 1
        if ex["correct"]:
            cluster_correct[ex["true_cluster"]] += 1

    per_cluster_acc = {
        cl: round(cluster_correct[cl] / cluster_total[cl], 4)
        for cl in sorted(cluster_total)
    }

    # Seen vs unseen accuracy
    seen_correct = sum(1 for ex in examples if ex["correct"] and ex["seen_during_training"])
    seen_total = sum(1 for ex in examples if ex["seen_during_training"])
    unseen_correct = sum(1 for ex in examples if ex["correct"] and not ex["seen_during_training"])
    unseen_total = sum(1 for ex in examples if not ex["seen_during_training"])

    # Confusion pairs: most common misclassifications
    confusion = Counter()
    for ex in examples:
        if not ex["correct"]:
            confusion[(ex["true_cluster"], ex["predicted_cluster"])] += 1

    return {
        "k": k,
        "total": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "seen_accuracy": round(seen_correct / seen_total, 4) if seen_total > 0 else None,
        "seen_total": seen_total,
        "unseen_accuracy": round(unseen_correct / unseen_total, 4) if unseen_total > 0 else None,
        "unseen_total": unseen_total,
        "per_cluster_accuracy": per_cluster_acc,
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in confusion.most_common(20)
        ],
        "examples": examples,
    }


# ============================================================================
# REPORTING
# ============================================================================

def print_report(results: dict):
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 80)
    print("  GNN CLUSTER EVALUATION REPORT")
    print("=" * 80)

    print(f"\n  KNN k={results['k']}")
    print(f"  Overall accuracy:  {results['accuracy']:.4f}  ({results['correct']}/{results['total']})")
    if results["seen_accuracy"] is not None:
        print(f"  Seen accuracy:     {results['seen_accuracy']:.4f}  (n={results['seen_total']})")
    if results["unseen_accuracy"] is not None:
        print(f"  Unseen accuracy:   {results['unseen_accuracy']:.4f}  (n={results['unseen_total']})")

    # Per-cluster accuracy
    print(f"\n  {'Cluster':<35} {'Acc':>8}  {'N':>4}")
    print("  " + "-" * 55)
    cluster_totals = defaultdict(int)
    for ex in results["examples"]:
        cluster_totals[ex["true_cluster"]] += 1
    for cl, acc in sorted(results["per_cluster_accuracy"].items(), key=lambda x: x[1]):
        n = cluster_totals[cl]
        bar = "#" * int(acc * 20)
        print(f"  {cl:<35} {acc:>8.4f}  {n:>4}  {bar}")

    # Top confusions
    if results["top_confusions"]:
        print(f"\n  Top misclassifications:")
        print(f"  {'True':<25} {'Predicted':<25} {'Count':>5}")
        print("  " + "-" * 60)
        for conf in results["top_confusions"][:15]:
            print(f"  {conf['true']:<25} {conf['predicted']:<25} {conf['count']:>5}")

    # Failed examples (sample)
    failures = [ex for ex in results["examples"] if not ex["correct"]]
    if failures:
        print(f"\n  Failed examples ({len(failures)} total, showing up to 30):")
        print(f"  {'Type':<5} {'True':<22} {'Pred':<22} {'Seen':>4}  Entity text")
        print("  " + "-" * 100)
        for ex in failures[:30]:
            seen_str = "Y" if ex["seen_during_training"] else "N"
            text = ex["entity_text"][:60]
            print(f"  {ex['entity_type']:<5} {ex['true_cluster']:<22} {ex['predicted_cluster']:<22} {seen_str:>4}  {text}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate spider model on eval_set_clustering.json")
    parser.add_argument("--model-dir", required=True, help="Path to stored_models/ directory")
    parser.add_argument("--eval-set", default=EVAL_SET_PATH, help="Path to eval_set_clustering.json")
    parser.add_argument("--k", type=int, default=5, help="KNN k value (default: 5)")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu (default: auto)")
    parser.add_argument("--output", default=None, help="Save full results to JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate UMAP visualization")
    parser.add_argument("--plot-dir", default="eval_spider_output", help="Output dir for plots")
    args = parser.parse_args()

    import torch

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load eval set
    with open(args.eval_set, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entities from {args.eval_set}")
    print(f"  Clusters: {len(set(e['cluster'] for e in dataset))}")
    print(f"  Seen: {sum(1 for e in dataset if e['seen_during_training'])}, "
          f"Unseen: {sum(1 for e in dataset if not e['seen_during_training'])}")

    # Load model & compute embeddings
    model, tokenizer, hidden_size = load_model_and_tokenizer(args.model_dir, device)
    embeddings = compute_embeddings(model, tokenizer, dataset, device, batch_size=args.batch_size)

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate
    results = evaluate(dataset, embeddings, k=args.k)
    print_report(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full results to {args.output}")

    # Plot
    if args.plot:
        plot_embeddings(embeddings, dataset, results, args.plot_dir)


def plot_embeddings(embeddings: np.ndarray, dataset: List[dict], results: dict, output_dir: str):
    """Generate UMAP scatter plots colored by cluster and correctness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    try:
        from umap import UMAP
        coords = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)).fit_transform(embeddings)

    clusters = [e["cluster"] for e in dataset]
    unique_clusters = sorted(set(clusters))
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))
    color_map = {c: i for i, c in enumerate(unique_clusters)}

    # Plot by cluster
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    for cl in unique_clusters:
        mask = [i for i, c in enumerate(clusters) if c == cl]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(color_map[cl])],
                   label=cl, s=25, alpha=0.8, edgecolors="none")
    ax.set_title("SPIDER Embeddings — by Cluster", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "by_cluster.png"), dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)

    # Plot by correctness
    examples = results["examples"]
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    correct_mask = [i for i, ex in enumerate(examples) if ex["correct"]]
    wrong_mask = [i for i, ex in enumerate(examples) if not ex["correct"]]
    ax.scatter(coords[correct_mask, 0], coords[correct_mask, 1], c="#22cc66",
               label=f"Correct ({len(correct_mask)})", s=20, alpha=0.6, edgecolors="none")
    ax.scatter(coords[wrong_mask, 0], coords[wrong_mask, 1], c="#ff3344",
               label=f"Wrong ({len(wrong_mask)})", s=40, alpha=0.9, edgecolors="white", linewidths=0.5)
    ax.set_title("SPIDER — KNN Prediction Correctness", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    legend = ax.legend(loc="upper right", fontsize=10, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "by_correctness.png"), dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)

    # Plot by seen/unseen
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    seen_mask = [i for i, e in enumerate(dataset) if e["seen_during_training"]]
    unseen_mask = [i for i, e in enumerate(dataset) if not e["seen_during_training"]]
    ax.scatter(coords[seen_mask, 0], coords[seen_mask, 1], c="#4488ff",
               label=f"Seen ({len(seen_mask)})", s=20, alpha=0.6, edgecolors="none")
    ax.scatter(coords[unseen_mask, 0], coords[unseen_mask, 1], c="#ffaa22",
               label=f"Unseen ({len(unseen_mask)})", s=30, alpha=0.8, edgecolors="none")
    ax.set_title("SPIDER — Seen vs Unseen", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    legend = ax.legend(loc="upper right", fontsize=10, frameon=False)
    for text in legend.get_texts():
        text.set_color("white")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "by_seen.png"), dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
