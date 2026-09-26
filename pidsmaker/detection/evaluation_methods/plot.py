from collections import defaultdict
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

def _precision_and_detected_attacks(scores, y_truth, nodes, node2attacks, thresholds_raw):
    """
    Compute precision(t) and detected_attack_rate(t) over thresholds.
    detected_attack_rate(t) = 1.0 iff every attack has at least one node predicted positive at threshold t.
    """
    scores = np.asarray(scores, dtype=float)
    y_truth = np.asarray(y_truth, dtype=int)
    thresholds_raw = np.asarray(thresholds_raw, dtype=float)

    # All distinct attacks in dataset
    attack_sets = list(node2attacks.values())
    all_attacks = set().union(*attack_sets) if len(attack_sets) > 0 else set()
    total_attacks = max(len(all_attacks), 1)  # avoid division by zero

    # Map attack -> indices of nodes belonging to that attack
    node_index = {n: i for i, n in enumerate(nodes)}
    attack2idxs = defaultdict(list)
    for n, atts in node2attacks.items():
        if n in node_index:
            idx = node_index[n]
            for a in atts:
                attack2idxs[a].append(idx)

    precision = np.zeros_like(thresholds_raw, dtype=float)
    det_rate  = np.zeros_like(thresholds_raw, dtype=float)

    for i, thr in enumerate(thresholds_raw):
        preds = (scores >= thr).astype(int)

        # Precision
        tp = np.sum((preds == 1) & (y_truth == 1))
        fp = np.sum((preds == 1) & (y_truth == 0))
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # All attacks detected?
        if len(all_attacks) == 0:
            det_rate[i] = 0.0
        else:
            detected = 0
            for a, idxs in attack2idxs.items():
                if idxs and np.any(preds[np.asarray(idxs, dtype=int)] == 1):
                    detected += 1
            det_rate[i] = detected / total_attacks

    return precision, det_rate


def plot_scores_distribution(
    scores, y_truth, nodes, node2attacks,
    out_file=None, threshold=None,
    n_curve_points=200, precision_cut=0.5,
    node_to_type=None,
):
    # Keep raw for thresholding; normalize only for plotting
    raw_scores = np.asarray(scores, dtype=float)
    y_truth = np.asarray(y_truth, dtype=int)
    raw_min, raw_max = float(np.min(raw_scores)), float(np.max(raw_scores))
    span = (raw_max - raw_min) if raw_max > raw_min else 1.0
    norm_scores = (raw_scores - raw_min) / span

    # Map any provided threshold to normalized space for drawing
    norm_threshold = None
    if threshold is not None:
        norm_threshold = (float(threshold) - raw_min) / span

    # Colors — three distinct shades of green for benign node types
    benign_type_colors = {
        "subject": "#1b5e20",  # dark forest green
        "file":    "#bbbbbb",  # mid green
        "netflow": "#a5d6a7",  # light mint green
    }
    benign_fallback = "#4daf4a"  # used when node_to_type is not provided
    attack_colors = {
        0: "black",
        1: "red",
        2: "#377eb8",
    }
    alpha_val = 0.7

    # Hist splits
    benign_by_type = defaultdict(list)
    attack_scores = {}
    for s_norm, label, node in zip(norm_scores, y_truth, nodes):
        if label == 0:
            if node_to_type is not None:
                ntype = node_to_type.get(node, "file")
            else:
                ntype = "all"
            benign_by_type[ntype].append(s_norm)
        else:
            attack_type = list(node2attacks.get(node))[0]
            attack_scores.setdefault(attack_type, []).append(s_norm)

    # Figure
    bins = np.linspace(0, 1, 75)
    plt.figure(figsize=(4.5, 3))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.grid(axis="x", visible=False)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Benign — one bar per node type (or single bar if no type info)
    legend_patches = []
    if node_to_type is not None:
        for ntype in ("subject", "file", "netflow"):
            vals = benign_by_type.get(ntype, [])
            if not vals:
                continue
            color = benign_type_colors[ntype]
            label = f"Benign ({ntype})"
            plt.hist(
                vals, bins=bins, alpha=alpha_val, label=label,
                color=color, edgecolor="black", linewidth=0.5, log=True
            )
            legend_patches.append(
                Patch(facecolor=color, edgecolor="black", alpha=alpha_val, label=label)
            )
    else:
        vals = benign_by_type.get("all", [])
        plt.hist(
            vals, bins=bins, alpha=alpha_val, label="Benign",
            color=benign_fallback, edgecolor="black", linewidth=0.5, log=True
        )
        legend_patches.append(
            Patch(facecolor=benign_fallback, edgecolor="black", alpha=alpha_val, label="Benign")
        )

    # Attacks
    for attack_type, values in attack_scores.items():
        plt.hist(
            values, bins=bins, alpha=alpha_val, label=f"Attack #{attack_type+1}",
            color=attack_colors.get(attack_type, "black"),
            edgecolor="black", linewidth=0.5, log=True
        )

    for atype in sorted(attack_scores.keys()):
        if atype in attack_colors:
            legend_patches.append(
                Patch(facecolor=attack_colors[atype], edgecolor="black", alpha=alpha_val, label=f"Attack #{atype+1}")
            )

    # ----- Precision/all-attacks-detected mask over thresholds -----
    thresholds_norm = np.linspace(0, 1, n_curve_points)        # for shading on the plot axis
    thresholds_raw  = raw_min + thresholds_norm * span         # for computations

    precision_curve, det_curve = _precision_and_detected_attacks(
        raw_scores, y_truth, nodes, node2attacks, thresholds_raw
    )

    # Shade regions where precision >= cut AND every attack detected (det_rate == 1)
    eps = 1e-12
    mask = (precision_curve >= precision_cut) & (det_curve >= 1.0 - eps)

    if np.any(mask):
        idx = np.where(mask)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)
        for run in runs:
            t_start = thresholds_norm[run[0]]
            t_end   = thresholds_norm[run[-1]]
            plt.axvspan(t_start, t_end, color="gray", alpha=0.2)
        # legend_patches.append(
        #     Patch(facecolor="gray", edgecolor="none", alpha=0.2,
        #           label=f"≥{int(precision_cut*100)}% precision, 100% detection")
        # )

    # Optional vertical threshold
    if norm_threshold is not None:
        plt.axvline(
            x=norm_threshold, color="black", linestyle="--", linewidth=1.5,
            label=f"Threshold: {norm_threshold:.2f}"
        )

    # Labels
    plt.xlabel("Node anomaly scores", fontsize=12)
    if out_file and "kairos" in out_file:
        plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if out_file and "kairos" in out_file:
        plt.legend(handles=legend_patches, frameon=True, fontsize=12)
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
