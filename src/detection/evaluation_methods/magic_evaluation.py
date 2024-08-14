from data_utils import *
from provnet_utils import *

from .evaluation_utils import compute_tw_labels_for_magic

from magic_utils import magic_eval
import os
import numpy as np
from .evaluation_utils import *
import torch
import wandb
from labelling import get_ground_truth

def transfer_result_to_node_evaluation(results, node_to_max_loss_tw):
    node_results = {}

    for tw, nid2data in results.items():
        node_results[tw] = {}

        for node_id, data in nid2data.items():
            if node_id not in node_results:
                node_results[node_id] = {}
                node_results[node_id]['y_true'] = 0
                node_results[node_id]['y_hat'] = 0

            node_results[node_id]['y_true'] = node_results[node_id]['y_true'] or data['y_true']
            node_results[node_id]['y_hat'] = node_results[node_id]['y_hat'] or data['y_hat']

    for node_id in node_results.keys():
        node_results[node_id]['tw_with_max_loss'] = node_to_max_loss_tw[node_id]
        node_results[node_id]['score'] = results[node_to_max_loss_tw[node_id]][node_id]['score']

    return node_results


def analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes):
    log(f"Analysis of false positives:")
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_truth, y_preds)) if pred and not true]
    malicious_tws = set(tw_to_malicious_nodes.keys())
    num_fps_in_malicious_tw = 0

    for i in fp_indices:
        is_in_malicious_tw = max_val_loss_tw[i] in malicious_tws
        num_fps_in_malicious_tw += int(is_in_malicious_tw)

        log(f"FP node {nodes[i]} -> max loss: {pred_scores[i]:.3f} | max TW: {max_val_loss_tw[i]} "
            f"| is malicious TW: " + (" ✅" if is_in_malicious_tw else " ❌"))

    fp_in_malicious_tw_ratio = num_fps_in_malicious_tw / len(fp_indices) if len(fp_indices) > 0 else float("nan")
    log(f"Percentage of FPs present in malicious TWs: {fp_in_malicious_tw_ratio:.3f}")
    return fp_in_malicious_tw_ratio

def main(cfg):
    tw_to_malicious_nodes = compute_tw_labels_for_magic(cfg)

    node_tw_results, node_to_max_loss_tw = magic_eval.get_node_predictions(cfg, tw_to_malicious_nodes)

    method = cfg.detection.evaluation.used_method.strip()
    if method == "magic_node_evaluation":
        results = transfer_result_to_node_evaluation(node_tw_results, node_to_max_loss_tw)
    else:
        log(f"Method {method} not supported.")

    node_to_path = get_node_to_path_and_type(cfg)

    model_epoch_dir = "magic_evaluation"

    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    dor_img_file = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")

    log("Analysis of malicious nodes:")
    nodes, y_truth, y_preds, pred_scores, max_val_loss_tw = [], [], [], [], []
    for nid, result in results.items():
        nodes.append(nid)
        score, y_hat, y_true, max_tw = result["score"], result["y_hat"], result["y_true"], result["tw_with_max_loss"]
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)
        max_val_loss_tw.append(max_tw)

        if y_true == 1:
            log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (
            node_to_path[nid]['path']))

    # Plots the PR curve and scores for mean node loss
    print(f"Saving figures to {out_dir}...")
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_dor_recall_curve(pred_scores, y_truth, dor_img_file)
    plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)
    plot_scores_with_paths(pred_scores, y_truth, nodes, max_val_loss_tw, tw_to_malicious_nodes, scores_img_file, cfg)
    stats = classifier_evaluation(y_truth, y_preds, pred_scores)

    fp_in_malicious_tw_ratio = analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes,
                                                       tw_to_malicious_nodes)
    stats["fp_in_malicious_tw_ratio"] = fp_in_malicious_tw_ratio

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")

    torch.save(results, results_file)
    torch.save(stats, stats_file)

    stats["precision_recall_img"] = wandb.Image(
        os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png"))
    stats["scores_img"] = wandb.Image(
        os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png"))

    wandb.log(stats)

    best_stats = stats
    wandb.log(best_stats)

    return stats


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)