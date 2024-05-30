from collections import defaultdict

import torch
import numpy as np
import wandb

from provnet_utils import *
from config import *
from .evaluation_utils import *


def get_node_predictions(val_tw_path, test_tw_path, cfg):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.node_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_losses = defaultdict(list)
    node_to_max_loss_tw = defaultdict(int)
    
    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                dstnode = data['dstnode']
                loss = data['loss']
                
                # Scores
                node_to_losses[srcnode].append(loss)
                if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    node_to_losses[dstnode].append(loss)
                    
                # If max-val thr is used, we want to keep track when the node with max loss happens
                if loss > node_to_max_loss_tw[srcnode]:
                    node_to_max_loss_tw[srcnode] = tw
                if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    if loss > node_to_max_loss_tw[dstnode]:
                        node_to_max_loss_tw[dstnode] = tw
                    
    results = defaultdict(dict)
    for node_id, losses in node_to_losses.items():
        pred_score = reduce_losses_to_score(losses, cfg.detection.evaluation.node_evaluation.threshold_method)

        results[node_id]["score"] = pred_score
        results[node_id]["y_hat"] = int(pred_score > thr)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["path"] = ground_truth_paths.get(node_id, "") # TODO: support for benign nodes too (see with Baowiang)
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw[node_id]

    return results

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

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results = get_node_predictions(val_tw_path, test_tw_path, cfg)

    os.makedirs(cfg.detection.evaluation.node_evaluation._precision_recall_dir, exist_ok=True)
    pr_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png")
    
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
            log(f"-> Malicious node {nid} ({result['path']}): loss={score:.3f} | is TP:" + (" ✅" if y_true == y_hat else " ❌"))

    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_scores(pred_scores, y_truth, scores_img_file)
    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    
    fp_in_malicious_tw_ratio = analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes)
    stats["fp_in_malicious_tw_ratio"] = fp_in_malicious_tw_ratio
    
    return stats
