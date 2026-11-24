"""Time-window level anomaly detection evaluation.

Aggregates edge-level scores to time-window level and classifies entire windows
as malicious or benign. Evaluates detection at coarser granularity than node-level,
useful for identifying suspicious time periods for investigation.
"""

import os

import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    get_threshold,
    load_losses_per_time_window,
    plot_precision_recall,
    plot_simple_scores,
    reduce_losses_to_score,
)
from pidsmaker.utils.utils import log


def get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes):
    """Compute time-window level predictions from edge (or node) losses.

    Args:
        val_tw_path: Validation results path (for threshold)
        test_tw_path: Test results path
        cfg: Configuration with threshold method
        tw_to_malicious_nodes: Ground truth time windows containing attacks

    Returns:
        tuple: (dict mapping TW index to {score, y_hat, y_true}, threshold)
    """
    threshold_method = cfg.evaluation.tw_evaluation.threshold_method
    thr = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {thr:.3f}")

    log(f"Loading data from {test_tw_path}...")
    _, tw_to_df = load_losses_per_time_window(test_tw_path, cfg)

    tw_labels = set(tw_to_malicious_nodes.keys())
    results = {}
    for tw, df in tw_to_df.items():
        pred_score = float(reduce_losses_to_score(df["loss"].tolist(), threshold_method))
        results[tw] = {
            "score": pred_score,
            "y_hat": int(pred_score > thr),
            "y_true": int(tw in tw_labels),
        }

    return results, thr


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results, thr = get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes)

    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")

    tws, y_truth, y_preds, pred_scores = [], [], [], []
    for tw, result in sorted(results.items()):
        score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
        tws.append(tw)
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)

        if y_true == 1:
            log(
                f"-> Malicious TW {tw}: loss={score:.3f} | is TP:"
                + (" ✅" if y_true == y_hat else " ❌")
            )

    if cfg.evaluation.save_figures is not False:  # unset: saved
        log(f"Saving figures to {out_dir}...")
        plot_precision_recall(pred_scores, y_truth, pr_img_file)
        plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)

    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    stats["threshold"] = thr

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
    torch.save(results, results_file)
    torch.save(stats, stats_file)
    torch.save(
        {"pred_scores": pred_scores, "y_preds": y_preds, "y_truth": y_truth, "tws": tws},
        scores_file,
    )
    stats["scores_file"] = scores_file
    return stats
