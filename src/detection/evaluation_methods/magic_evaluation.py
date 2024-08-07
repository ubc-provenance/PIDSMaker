from data_utils import *
from provnet_utils import *

from .evaluation_utils import compute_tw_labels_for_magic

from magic_utils import magic_eval
import os
import numpy as np
from .evaluation_utils import *
import torch
import wandb


def main(cfg):
    tw_to_malicious_nodes = compute_tw_labels_for_magic(cfg)

    results, node_to_max_loss_tw = magic_eval.get_node_predictions(cfg, tw_to_malicious_nodes)

    model_epoch_dir = "magic_evaluation"

    node_to_path = get_node_to_path_and_type(cfg)
    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")

    log("Analysis of malicious nodes:")
    nodes, y_truth, y_preds, pred_scores = [], [], [], []
    node_to_correct_pred = {}

    for tw, nid_to_result in results.items():
        malicious_tws = set()
        malicious_nodes = set()

        # We create a new arrayfor each TW
        for arr in [nodes, y_truth, y_preds, pred_scores]:
            arr.append([])
        for nid, result in nid_to_result.items():
            nodes[tw].append(nid)
            score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
            y_truth[tw].append(y_true)
            y_preds[tw].append(y_hat)
            pred_scores[tw].append(score)
            node_to_correct_pred[nid] = y_hat == y_true

            if y_true == 1:
                if tw not in malicious_tws:
                    log(f"TW {tw}")
                    malicious_tws.add(tw)
                log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (
                node_to_path[nid]['path']))
                malicious_nodes.add(nid)


    flat_pred_scores = [e for sublist in pred_scores for e in sublist]
    flat_y_truth = [e for sublist in y_truth for e in sublist]
    flat_y_preds = [e for sublist in y_preds for e in sublist]
    flat_nodes = [e for sublist in nodes for e in sublist]

    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(flat_pred_scores, flat_y_truth, pr_img_file)

    max_val_loss_tw = [node_to_max_loss_tw[n] for n in flat_nodes]
    plot_scores_with_paths(flat_pred_scores, flat_y_truth, flat_nodes, max_val_loss_tw, tw_to_malicious_nodes,
                           scores_img_file, cfg)
    stats = classifier_evaluation(flat_y_truth, flat_y_preds, flat_pred_scores)

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")

    torch.save(results, results_file)
    torch.save(stats, stats_file)

    stats["epoch"] = cfg.featurization.embed_edges.magic.max_epoch
    stats["precision_recall_img"] = wandb.Image(
        os.path.join(out_dir, f"{model_epoch_dir}.png"))
    stats["scores_img"] = wandb.Image(
        os.path.join(out_dir, f"scores_{model_epoch_dir}.png"))

    wandb.log(stats)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)