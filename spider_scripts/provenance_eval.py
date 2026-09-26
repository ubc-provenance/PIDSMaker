#!/usr/bin/env python3
"""
Provenance Foundation Model — Evaluation Suite

This module provides:
1. A comprehensive evaluation dataset (get_evaluation_dataset())
2. Metric computation (evaluate_embeddings())
3. Visualization (plot_evaluation_report())

Usage:
    from provenance_eval import get_evaluation_dataset, evaluate_embeddings, plot_evaluation_report

    # 1. Get the evaluation entities
    dataset = get_evaluation_dataset()
    
    # 2. Compute embeddings with your model
    embeddings = {}
    for entry in dataset:
        key = f"{entry['entity_type']}:{entry['entity_text']}"
        embeddings[key] = your_model.encode(entry['entity_type'], entry['entity_text'])
    
    # 3. Evaluate
    results = evaluate_embeddings(embeddings, dataset)
    
    # 4. Visualize
    plot_evaluation_report(embeddings, dataset, results, output_dir="eval_output/")
"""

import sys
import json
import os
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


# ============================================================================
# EVALUATION DATASET — loaded from eval_set.json
# ============================================================================

_EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval_set.json')

# Cache
_DATASET = None

def get_evaluation_dataset() -> List[dict]:
    """Return the evaluation dataset loaded from eval_set.json. Cached after first call."""
    global _DATASET
    if _DATASET is None:
        with open(_EVAL_SET_PATH, 'r') as f:
            _DATASET = json.load(f)
    return _DATASET


def get_dataset_stats() -> dict:
    """Return summary statistics about the evaluation dataset."""
    ds = get_evaluation_dataset()
    clusters = Counter(d['cluster'] for d in ds)
    tiers = Counter(d['tier'] for d in ds)
    oses = Counter(d['os'] for d in ds)
    etypes = Counter(d['entity_type'] for d in ds)
    xos_groups = set(d['xos_group'] for d in ds if d['xos_group'])
    
    return {
        'total_entities': len(ds),
        'n_clusters': len(clusters),
        'n_tiers': len(tiers),
        'n_xos_groups': len(xos_groups),
        'by_os': dict(oses),
        'by_type': dict(etypes),
        'by_tier': dict(tiers),
        'by_cluster': dict(clusters.most_common()),
    }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def knn_accuracy_in_full_graph(X_all, labeled_indices, labels, k=5):
    """
    X_all: all embeddings (millions)
    labeled_indices: which rows in X_all are labeled probes
    labels: cluster labels for those probes
    """
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter

    labeled_set = set(labeled_indices)

    # Search enough neighbors to find k labeled ones
    nn = NearestNeighbors(n_neighbors=min(200, len(X_all))).fit(X_all)

    correct = 0
    total = 0

    for i, idx in enumerate(labeled_indices):
        distances, indices = nn.kneighbors(X_all[idx:idx+1])

        # Collect k labeled neighbors (excluding self)
        voter_labels = []
        for j in indices[0]:
            if j in labeled_set and j != idx:
                j_label_pos = labeled_indices.index(j)
                voter_labels.append(labels[j_label_pos])
            if len(voter_labels) >= k:
                break

        if not voter_labels:
            continue

        prediction = Counter(voter_labels).most_common(1)[0][0]
        if prediction == labels[i]:
            correct += 1
        total += 1

    return correct / total


def knn_accuracy_per_cluster(X_all, labeled_indices, labels, k=5):
    """Run KNN over the full graph, then break down accuracy by true cluster label."""
    from sklearn.neighbors import NearestNeighbors
    from collections import Counter, defaultdict

    labeled_set = set(labeled_indices)
    nn = NearestNeighbors(n_neighbors=min(200, len(X_all))).fit(X_all)

    cluster_correct = defaultdict(int)
    cluster_total = defaultdict(int)

    for i, idx in enumerate(labeled_indices):
        distances, indices = nn.kneighbors(X_all[idx:idx+1])

        voter_labels = []
        for j in indices[0]:
            if j in labeled_set and j != idx:
                j_label_pos = labeled_indices.index(j)
                voter_labels.append(labels[j_label_pos])
            if len(voter_labels) >= k:
                break

        if not voter_labels:
            continue

        true_label = labels[i]
        prediction = Counter(voter_labels).most_common(1)[0][0]
        cluster_total[true_label] += 1
        if prediction == true_label:
            cluster_correct[true_label] += 1

    return {
        cl: round(cluster_correct[cl] / cluster_total[cl], 4)
        for cl in cluster_total
    }


def evaluate_embeddings(
    embeddings: Dict[str, np.ndarray],
    dataset: Optional[List[dict]] = None,
) -> dict:
    """
    Evaluate embedding quality against the evaluation dataset.
    
    Args:
        embeddings: dict mapping "ENTITY_TYPE:entity_text" → np.ndarray embedding vector
                    e.g. {"PROC:/usr/sbin/nginx": np.array([0.1, 0.2, ...]), ...}
        dataset: optional, defaults to get_evaluation_dataset()
    
    Returns:
        dict of metric names → values, with interpretive thresholds
    """
    if dataset is None:
        dataset = get_evaluation_dataset()
    
    # Match dataset entries to provided embeddings
    matched = []
    missing = []
    for entry in dataset:
        key = f"{entry['entity_type']}:{entry['entity_text']}"
        if key in embeddings:
            matched.append((entry, embeddings[key]))
        else:
            missing.append(key)
    
    if len(matched) < 10:
        raise ValueError(f"Only {len(matched)} entities matched. Provide embeddings with keys like 'PROC:/usr/sbin/nginx'. Missing: {missing[:5]}...")
    
    print(f"Matched {len(matched)}/{len(dataset)} entities ({len(missing)} missing)")
    
    X = np.stack([emb for _, emb in matched])
    entries = [e for e, _ in matched]
    
    clusters = [e['cluster'] for e in entries]
    tiers = [e['tier'] for e in entries]
    oses = [e['os'] for e in entries]
    etypes = [e['entity_type'] for e in entries]
    sub_clusters = [e.get('sub_cluster', '') for e in entries]
    xos_groups = [e.get('xos_group', None) for e in entries]
    
    results = {}
    
    # ── 1. Adjusted Rand Index ──
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    
    unique_clusters = list(set(clusters))
    k = len(unique_clusters)
    pred = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    ari = adjusted_rand_score(clusters, pred)
    results['adjusted_rand_index'] = round(ari, 4)
    
    # ── 2. Silhouette Score ──
    from sklearn.metrics import silhouette_score
    
    label_map = {c: i for i, c in enumerate(unique_clusters)}
    numeric_labels = [label_map[c] for c in clusters]
    sil = silhouette_score(X, numeric_labels)
    results['silhouette_score'] = round(sil, 4)
    
    # ── 3. Retrieval Precision@k ──
    from sklearn.neighbors import NearestNeighbors
    
    for k_val in [5, 10, 20]:
        nn_k = min(k_val + 1, len(X))
        nn = NearestNeighbors(n_neighbors=nn_k).fit(X)
        distances, indices = nn.kneighbors(X)
        
        precisions = []
        for i in range(len(X)):
            neighbors = indices[i][1:]  # exclude self
            same = sum(1 for j in neighbors if clusters[j] == clusters[i])
            precisions.append(same / len(neighbors))
        results[f'retrieval_precision_at_{k_val}'] = round(np.mean(precisions), 4)
    
    # ── 4. Tier-level Retrieval Precision@10 ──
    nn = NearestNeighbors(n_neighbors=min(11, len(X))).fit(X)
    distances, indices = nn.kneighbors(X)
    
    tier_precs = []
    for i in range(len(X)):
        neighbors = indices[i][1:]
        same = sum(1 for j in neighbors if tiers[j] == tiers[i])
        tier_precs.append(same / len(neighbors))
    results['tier_retrieval_precision_at_10'] = round(np.mean(tier_precs), 4)
    
    # ── 5. Cross-OS Retrieval ──
    xos_hits = 0
    xos_total = 0
    for i in range(len(X)):
        if xos_groups[i] is None:
            continue
        my_os = oses[i]
        my_group = xos_groups[i]
        for j in indices[i][1:]:
            other_os = oses[j]
            if other_os != my_os and other_os != 'cross':
                xos_total += 1
                if xos_groups[j] == my_group:
                    xos_hits += 1
                break
    
    results['cross_os_retrieval_accuracy'] = round(xos_hits / max(xos_total, 1), 4)
    results['cross_os_pairs_evaluated'] = xos_total
    
    # ── 6. Hierarchical Triplet Accuracy ──
    triplet_correct = 0
    triplet_total = 0
    rng = np.random.RandomState(42)
    
    for _ in range(min(5000, len(X) * 10)):
        anchor_idx = rng.randint(len(X))
        anchor_cluster = clusters[anchor_idx]
        anchor_tier = tiers[anchor_idx]
        
        # Positive: same cluster
        same_cluster_idxs = [j for j in range(len(X)) if clusters[j] == anchor_cluster and j != anchor_idx]
        if not same_cluster_idxs:
            continue
        
        # Negative: different cluster
        diff_cluster_idxs = [j for j in range(len(X)) if clusters[j] != anchor_cluster]
        if not diff_cluster_idxs:
            continue
        
        pos_idx = rng.choice(same_cluster_idxs)
        neg_idx = rng.choice(diff_cluster_idxs)
        
        d_pos = np.linalg.norm(X[anchor_idx] - X[pos_idx])
        d_neg = np.linalg.norm(X[anchor_idx] - X[neg_idx])
        
        triplet_total += 1
        if d_pos < d_neg:
            triplet_correct += 1
    
    results['triplet_accuracy'] = round(triplet_correct / max(triplet_total, 1), 4)
    results['triplets_evaluated'] = triplet_total
    
    # ── 7. Tier Triplet Accuracy (coarser hierarchy) ──
    tier_triplet_correct = 0
    tier_triplet_total = 0
    
    for _ in range(min(5000, len(X) * 10)):
        anchor_idx = rng.randint(len(X))
        anchor_tier = tiers[anchor_idx]
        
        same_tier_idxs = [j for j in range(len(X)) if tiers[j] == anchor_tier and j != anchor_idx]
        diff_tier_idxs = [j for j in range(len(X)) if tiers[j] != anchor_tier]
        if not same_tier_idxs or not diff_tier_idxs:
            continue
        
        pos_idx = rng.choice(same_tier_idxs)
        neg_idx = rng.choice(diff_tier_idxs)
        
        d_pos = np.linalg.norm(X[anchor_idx] - X[pos_idx])
        d_neg = np.linalg.norm(X[anchor_idx] - X[neg_idx])
        
        tier_triplet_total += 1
        if d_pos < d_neg:
            tier_triplet_correct += 1
    
    results['tier_triplet_accuracy'] = round(tier_triplet_correct / max(tier_triplet_total, 1), 4)
    
    # ── 8. Entity Type Mixedness ──
    cluster_types = defaultdict(set)
    for i in range(len(entries)):
        cluster_types[clusters[i]].add(etypes[i])
    
    mixed = sum(1 for types in cluster_types.values() if len(types) >= 2)
    results['type_mixed_clusters'] = f"{mixed}/{len(cluster_types)}"
    results['type_mixed_fraction'] = round(mixed / max(len(cluster_types), 1), 4)
    
    # ── 9. Attack vs Benign Separation ──
    attack_idxs = [i for i in range(len(X)) if tiers[i] == 'attack']
    benign_idxs = [i for i in range(len(X)) if tiers[i] != 'attack']
    
    if attack_idxs and benign_idxs:
        attack_center = X[attack_idxs].mean(axis=0)
        benign_center = X[benign_idxs].mean(axis=0)
        
        inter_dist = np.linalg.norm(attack_center - benign_center)
        
        attack_intra = np.mean([np.linalg.norm(X[i] - attack_center) for i in attack_idxs])
        benign_intra = np.mean([np.linalg.norm(X[i] - benign_center) for i in benign_idxs])
        
        # Ratio > 1 means clusters are separated; > 2 is strong separation
        separation = inter_dist / (0.5 * (attack_intra + benign_intra) + 1e-10)
        results['attack_benign_separation'] = round(separation, 4)
    
    # ── 10. KNN Accuracy in Full Graph ──
    all_indices = list(range(len(X)))
    knn_acc = knn_accuracy_in_full_graph(X, all_indices, clusters, k=5)
    results['knn_accuracy_k5'] = round(knn_acc, 4)

    # Per-cluster KNN accuracy (using full-graph predictions, grouped by true label)
    knn_per_cluster = knn_accuracy_per_cluster(X, all_indices, clusters, k=5)
    results['knn_accuracy_k5_per_cluster'] = knn_per_cluster

    # ── 11. Per-cluster stats ──
    per_cluster = {}
    for cl in unique_clusters:
        cl_idxs = [i for i in range(len(X)) if clusters[i] == cl]
        if len(cl_idxs) < 2:
            continue
        cl_X = X[cl_idxs]
        centroid = cl_X.mean(axis=0)
        intra_dist = np.mean(np.linalg.norm(cl_X - centroid, axis=1))
        per_cluster[cl] = {
            'count': len(cl_idxs),
            'mean_intra_distance': round(float(intra_dist), 4),
            'os_distribution': dict(Counter(oses[i] for i in cl_idxs)),
        }
    results['per_cluster'] = per_cluster
    
    # ── Thresholds for interpretation ──
    results['_thresholds'] = {
        'adjusted_rand_index': {'bad': '<0.1', 'okay': '0.1-0.3', 'good': '0.3-0.6', 'great': '>0.6'},
        'silhouette_score': {'bad': '<0', 'okay': '0-0.15', 'good': '0.15-0.35', 'great': '>0.35'},
        'retrieval_precision_at_10': {'bad': '<0.2', 'okay': '0.2-0.5', 'good': '0.5-0.7', 'great': '>0.7'},
        'cross_os_retrieval_accuracy': {'bad': '<0.1', 'okay': '0.1-0.3', 'good': '0.3-0.6', 'great': '>0.6'},
        'triplet_accuracy': {'bad': '<0.6', 'okay': '0.6-0.75', 'good': '0.75-0.9', 'great': '>0.9'},
        'attack_benign_separation': {'bad': '<0.5', 'okay': '0.5-1.0', 'good': '1.0-2.0', 'great': '>2.0'},
        'knn_accuracy_k5': {'bad': '<0.5', 'okay': '0.5-0.7', 'good': '0.7-0.85', 'great': '>0.85'},
    }
    
    return results


def print_evaluation_report(results: dict):
    """Pretty-print the evaluation results."""
    thresholds = results.get('_thresholds', {})
    
    print("\n" + "=" * 70)
    print("  PROVENANCE EMBEDDING EVALUATION REPORT")
    print("=" * 70)
    
    def grade(metric_name, value):
        if metric_name not in thresholds:
            return ""
        t = thresholds[metric_name]
        if 'great' in t:
            great_val = float(t['great'].replace('>', ''))
            good_val = float(t['good'].split('-')[0])
            okay_val = float(t['okay'].split('-')[0])
            if value >= great_val: return "★★★ GREAT"
            if value >= good_val: return "★★  GOOD"
            if value >= okay_val: return "★   OKAY"
            return "    BAD"
        return ""
    
    metrics = [
        ('adjusted_rand_index', 'Adjusted Rand Index'),
        ('silhouette_score', 'Silhouette Score'),
        ('retrieval_precision_at_5', 'Retrieval P@5'),
        ('retrieval_precision_at_10', 'Retrieval P@10'),
        ('retrieval_precision_at_20', 'Retrieval P@20'),
        ('tier_retrieval_precision_at_10', 'Tier Retrieval P@10'),
        ('cross_os_retrieval_accuracy', f"Cross-OS Retrieval ({results.get('cross_os_pairs_evaluated', '?')} pairs)"),
        ('triplet_accuracy', f"Triplet Accuracy ({results.get('triplets_evaluated', '?')} triplets)"),
        ('tier_triplet_accuracy', 'Tier Triplet Accuracy'),
        ('knn_accuracy_k5', 'KNN Accuracy (k=5)'),
        ('attack_benign_separation', 'Attack/Benign Separation'),
        ('type_mixed_fraction', 'Type-Mixed Clusters'),
    ]
    
    print(f"\n{'Metric':<45} {'Value':>8}  {'Grade'}")
    print("-" * 70)
    for key, name in metrics:
        if key in results:
            val = results[key]
            g = grade(key, val)
            print(f"  {name:<43} {val:>8.4f}  {g}")
    
    if 'type_mixed_clusters' in results:
        print(f"\n  Type-mixed clusters: {results['type_mixed_clusters']}")

    # Per-cluster KNN accuracy breakdown
    if 'knn_accuracy_k5_per_cluster' in results:
        print(f"\n  KNN Accuracy (k=5) per cluster:")
        for cl, acc in sorted(results['knn_accuracy_k5_per_cluster'].items()):
            g = grade('knn_accuracy_k5', acc)
            print(f"    {cl:<40} {acc:>8.4f}  {g}")

    # Per-cluster breakdown
    if 'per_cluster' in results:
        print(f"\n{'Cluster':<30} {'Count':>5}  {'Intra-dist':>10}  {'OS mix'}")
        print("-" * 70)
        for cl, stats in sorted(results['per_cluster'].items()):
            os_str = ', '.join(f"{k}:{v}" for k, v in stats['os_distribution'].items())
            print(f"  {cl:<28} {stats['count']:>5}  {stats['mean_intra_distance']:>10.4f}  {os_str}")
    
    print("\n" + "=" * 70)


def plot_evaluation_report(
    embeddings: Dict[str, np.ndarray],
    dataset: Optional[List[dict]] = None,
    results: Optional[dict] = None,
    output_dir: str = "eval_output",
):
    """Generate UMAP plots colored by cluster, tier, OS, and entity type."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if dataset is None:
        dataset = get_evaluation_dataset()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Match
    matched = []
    for entry in dataset:
        key = f"{entry['entity_type']}:{entry['entity_text']}"
        if key in embeddings:
            matched.append((entry, embeddings[key]))
    
    X = np.stack([emb for _, emb in matched])
    entries = [e for e, _ in matched]
    
    # 2D reduction
    try:
        from umap import UMAP
        coords = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1)).fit_transform(X)
    
    def make_plot(color_key, title, filename):
        values = [e[color_key] for e in entries]
        unique_vals = sorted(set(values))
        color_map = {v: i for i, v in enumerate(unique_vals)}
        cmap = plt.cm.get_cmap('tab20', len(unique_vals))
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        
        for val in unique_vals:
            mask = [i for i, v in enumerate(values) if v == val]
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      c=[cmap(color_map[val])], label=val, s=30, alpha=0.8, edgecolors='none')
        
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        
        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                          fontsize=7, ncol=1, frameon=False)
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight',
                   facecolor='#0a0a0f')
        plt.close(fig)
        print(f"  Saved {filename}")
    
    make_plot('cluster', 'Embeddings by Functional Cluster', 'by_cluster.png')
    make_plot('tier', 'Embeddings by Tier', 'by_tier.png')
    make_plot('os', 'Embeddings by OS', 'by_os.png')
    make_plot('entity_type', 'Embeddings by Entity Type', 'by_entity_type.png')
    
    # Save coords for interactive viewer
    coords_data = []
    for i, (entry, _) in enumerate(matched):
        coords_data.append({
            'x': round(float(coords[i, 0]), 4),
            'y': round(float(coords[i, 1]), 4),
            't': entry['entity_type'],
            'text': entry['entity_text'],
            'c': entry['cluster'],
            's': entry.get('sub_cluster', ''),
            'os': entry['os'],
            'tier': entry['tier'],
        })
    with open(os.path.join(output_dir, 'points.json'), 'w') as f:
        json.dump(coords_data, f, separators=(',', ':'))
    print(f"  Saved points.json (for interactive viewer)")


# ============================================================================
# EMBEDDING COMPUTATION
# ============================================================================

# Maps eval dataset entity_type to the node types used by the tokenizer
EVAL_TYPE_MAP = {'PROC': 'subject', 'FILE': 'file', 'SOCK': 'netflow', 'NETFLOW': 'netflow'}


def compute_embeddings(
    model_dir: str,
    dataset: Optional[List[dict]] = None,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for the evaluation dataset using a trained T5 model.

    Args:
        model_dir: Path to stored_models/ directory containing tokenizer.pt and best_mini.pt
        dataset: optional, defaults to get_evaluation_dataset()
        batch_size: encoding batch size
        device: 'cuda', 'cpu', or None for auto-detect

    Returns:
        dict mapping "ENTITY_TYPE:entity_text" → np.ndarray embedding vector
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pidsmaker.spider.models.t5 import ProvenanceT5, get_t5_config
    from pidsmaker.spider.data.tokenizer_bpe import ProvenanceTokenizerBPE

    if dataset is None:
        dataset = get_evaluation_dataset()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = object.__new__(ProvenanceTokenizerBPE)
    tokenizer.expand_netflow_ips = False
    tokenizer.load(os.path.join(model_dir, 'tokenizer.pt'))

    # Load model
    model_path = os.path.join(model_dir, 'pretrain_mini.pt')
    model_state = torch.load(model_path, map_location='cpu', weights_only=False)
    config = get_t5_config(tokenizer.vocab_size, 'mini')
    model = ProvenanceT5(config)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device).eval()
    print(f"Loaded model from {model_dir}")

    # Tokenize all entities
    node_types = [EVAL_TYPE_MAP[e['entity_type']] for e in dataset]
    labels = [e['entity_text'] for e in dataset]

    all_token_ids = []
    for ntype, label in zip(node_types, labels):
        ids = tokenizer.tokenize_node(ntype, label)
        if not ids:
            ids = [tokenizer.pad_id]
        all_token_ids.append(ids)

    # Encode in batches
    n_total = len(all_token_ids)
    all_embeddings = []
    with torch.no_grad():
        for batch_start in range(0, n_total, batch_size):
            if batch_start % (batch_size * 20) == 0:
                print(f"  Encoding {batch_start}/{n_total}...")
            batch = all_token_ids[batch_start:batch_start + batch_size]
            max_len = min(max(len(t) for t in batch), tokenizer.max_seq_len)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
            for i, tids in enumerate(batch):
                seq_len = min(len(tids), max_len)
                input_ids[i, :seq_len] = torch.tensor(tids[:seq_len])
                attention_mask[i, :seq_len] = True
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            hidden_states = model.modified_fwd(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=torch.full_like(input_ids, -100), skip_cls=True)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mean_pooled = ((hidden_states * mask_expanded).sum(dim=1) /
                           mask_expanded.sum(dim=1).clamp(min=1)).cpu().numpy()
            all_embeddings.append(mean_pooled)

    emb = np.concatenate(all_embeddings, axis=0)
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb = emb / norms

    print(f"  Encoded {n_total} entities → {emb.shape[1]}D embeddings")

    # Build the embeddings dict keyed by "ENTITY_TYPE:entity_text"
    embeddings = {}
    for i, entry in enumerate(dataset):
        key = f"{entry['entity_type']}:{entry['entity_text']}"
        embeddings[key] = emb[i]

    return embeddings


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Provenance Embedding Evaluation')
    parser.add_argument('--model-dir', help='Path to stored_models/ dir (computes embeddings directly)')
    parser.add_argument('--embeddings', help='Path to embeddings file (JSON: {"key": [vec]})')
    parser.add_argument('--stats', action='store_true', help='Print dataset statistics')
    parser.add_argument('--export', help='Export evaluation dataset to JSON file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', default='eval_output', help='Output directory for plots')
    parser.add_argument('--save-embeddings', help='Save computed embeddings to JSON file')
    parser.add_argument('--device', default=None, help='Device: cuda, cpu (default: auto)')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for encoding')
    args = parser.parse_args()

    if args.stats:
        stats = get_dataset_stats()
        print(json.dumps(stats, indent=2))

    if args.export:
        ds = get_evaluation_dataset()
        with open(args.export, 'w') as f:
            json.dump(ds, f, indent=2)
        print(f"Exported {len(ds)} entities to {args.export}")

    embeddings = None
    ds = get_evaluation_dataset()

    if args.model_dir:
        embeddings = compute_embeddings(
            args.model_dir, ds,
            batch_size=args.batch_size, device=args.device)
        if args.save_embeddings:
            save_data = {k: v.tolist() for k, v in embeddings.items()}
            with open(args.save_embeddings, 'w') as f:
                json.dump(save_data, f)
            print(f"Saved embeddings to {args.save_embeddings}")
    elif args.embeddings:
        with open(args.embeddings) as f:
            embeddings = {k: np.array(v) for k, v in json.load(f).items()}

    if embeddings is not None:
        results = evaluate_embeddings(embeddings, ds)
        print_evaluation_report(results)
        if args.plot:
            plot_evaluation_report(embeddings, ds, results, args.output_dir)

    if not any([args.stats, args.export, args.embeddings, args.model_dir]):
        stats = get_dataset_stats()
        print(f"Evaluation dataset: {stats['total_entities']} entities")
        print(f"  Clusters: {stats['n_clusters']}")
        print(f"  Tiers: {stats['n_tiers']}")
        print(f"  Cross-OS groups: {stats['n_xos_groups']}")
        print(f"  By OS: {stats['by_os']}")
        print(f"  By type: {stats['by_type']}")
        print(f"\nUsage:")
        print(f"  python provenance_eval.py --stats")
        print(f"  python provenance_eval.py --export eval_dataset.json")
        print(f"  python provenance_eval.py --model-dir /path/to/stored_models")
        print(f"  python provenance_eval.py --model-dir /path/to/stored_models --plot")
        print(f"  python provenance_eval.py --model-dir /path/to/stored_models --save-embeddings emb.json")
        print(f"  python provenance_eval.py --embeddings my_embeddings.json")