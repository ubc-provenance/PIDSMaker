from collections import defaultdict

import torch
import numpy as np
import wandb

from provnet_utils import *
from config import *
from .evaluation_utils import *


def get_node_thr(val_tw_path, cfg):
    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method
    if threshold_method == "max_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path)['max']
    # elif threshold_method == "supervised_best_threshold":
        # thr = calculate_supervised_best_threshold(losses, edge_labels)
    elif threshold_method == "avg_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path)['avg']
    elif threshold_method == "90_percent_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path)['percentile_90']
    else:
        raise ValueError(f"Invalid threshold method `{threshold_method}`")
    
    return thr

def get_node_scores(tw_path, cfg):
    ground_truth_nids = get_ground_truth_nids(cfg)
    log(f"Loading data from {tw_path}...")

    edge_index, losses, item_to_scores = [], [], defaultdict(list)
    labels = {}
    
    filelist = listdir_sorted(tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(tw_path, file)
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                dstnode = data['dstnode']
                loss = data['loss']
                
                edge_index.append((srcnode, dstnode))
                losses.append(loss)
                # Labels
                labels[srcnode] = 0
                if dstnode not in labels and cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    labels[dstnode] = 0
                # Scores
                item_to_scores[srcnode].append(loss)
                if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    item_to_scores[dstnode].append(loss)

    for nid in labels:
        if nid in ground_truth_nids:
            labels[nid] = 1

    return edge_index, losses, labels, item_to_scores

def main(val_tw_path, tw_path, model_epoch_dir, cfg, **kwargs):
    edge_index, losses, node_labels, item_to_scores = get_node_scores(tw_path, cfg)

    os.makedirs(cfg.detection.evaluation.node_evaluation._precision_recall_dir, exist_ok=True)
    pr_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png")
    
    thr = get_node_thr(val_tw_path, cfg) # TODO: change as it only works for max, not for mean (need to do the mean of each tw)
    log(f"Threshold: {thr:.3f}")
    
    y_truth, y_pred, pred_scores = [], [], []
    for nid, loss_list in item_to_scores.items():
        pred_score = None
        if cfg.detection.evaluation.node_evaluation.use_mean_node_loss:
            pred_score = np.mean(loss_list)
        else:
            pred_score = np.max(loss_list)

        y_truth.append(node_labels[nid])
        y_pred.append(int(pred_score > thr))
        pred_scores.append(pred_score)
        if node_labels[nid]:
            log(f"-> Malicious node {nid}: loss={pred_score:.3f}" + (" ✅" if pred_score > thr else " ❌"))
            
    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_scores(pred_scores, y_truth, scores_img_file)
    return classifier_evaluation(y_truth, y_pred, pred_scores)
