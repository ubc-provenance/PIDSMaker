from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import re
import wandb

from provnet_utils import *
from config import *


def calculate_max_val_loss_threshold(graph_dir, logger):
    filelist = listdir_sorted(graph_dir)

    loss_list = []
    for file in sorted(filelist):
        f = open(os.path.join(graph_dir, file))
        for line in f:
            l = line.strip()
            jdata = eval(l)

            loss_list.append(jdata['loss'])

    thr = {
        'max': max(loss_list),
        'avg': mean(loss_list),
        'percentile_90': percentile_90(loss_list)
           }
    # thr = np.percentile(loss_list, 90)
    logger.info(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

    return thr

def calculate_supervised_best_threshold(losses, labels):
    fpr, tpr, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpr, tpr)

    valid_indices = np.where(tpr >= 0.16)[0]
    fpr_valid = fpr[valid_indices]
    thresholds_valid = thresholds[valid_indices]

    # Find the threshold corresponding to the lowest FPR among valid points
    optimal_idx = np.argmin(fpr_valid)
    optimal_threshold = thresholds_valid[optimal_idx]

    return optimal_threshold

def get_ground_truth_nids(cfg):
    ground_truth_nids = []
    with open(os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path_new), 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
    return ground_truth_nids

def get_edge_index_losses_labels(tw_path, logger, cfg):
    ground_truth_nids = get_ground_truth_nids(cfg)

    logger.info(f"Loading data from {tw_path}...")
    node_labels = {}
    edge_index, losses, edge_labels, node2mean_loss = [], [], [], defaultdict(list)

    filelist = listdir_sorted(tw_path)
    for file in tqdm(sorted(filelist), desc="Compute labels"):
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

                if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    edge_labels.append(int(srcnode in ground_truth_nids or dstnode in ground_truth_nids))
                else:
                    edge_labels.append(int(srcnode in ground_truth_nids))

                if srcnode not in node_labels:
                    node_labels[srcnode] = 0
                if dstnode not in node_labels and cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    node_labels[dstnode] = 0

                node2mean_loss[srcnode].append(loss)
                if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                    node2mean_loss[dstnode].append(loss)

    for nid in node_labels:
        if nid in ground_truth_nids:
            node_labels[nid] = 1

    for nid, loss_list in node2mean_loss.items():
        node2mean_loss[nid] = np.mean(loss_list)

    return edge_index, losses, node_labels, edge_labels, node2mean_loss

def node_evaluation_without_triage(val_tw_path, tw_path, model_epoch_dir, logger, cfg):
    edge_index, losses, node_labels, edge_labels, node2mean_loss = get_edge_index_losses_labels(tw_path, logger, cfg)

    os.makedirs(cfg.detection.evaluation.node_evaluation._precision_recall_dir, exist_ok=True)
    img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png")
    # plot_precision_recall(edge_labels, losses, img_file)

    node_mean, node_mean_labels = [], []
    for nid, mean in node2mean_loss.items():
        node_mean.append(mean)
        node_mean_labels.append(node_labels[nid])
    plot_precision_recall(node_mean_labels, node_mean, img_file)
    
    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method
    if threshold_method == "max_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path, logger)['max']
    elif threshold_method == "supervised_best_threshold":
        thr = calculate_supervised_best_threshold(losses, edge_labels)
    elif threshold_method == "avg_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path, logger)['avg']
    elif threshold_method == "90_percent_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path, logger)['percentile_90']
    else:
        raise ValueError(f"Invalid threshold method `{threshold_method}`")
    
    y_truth, y_pred = [], []
    # Thresholds each node based on the mean of the losses of this node
    if cfg.detection.evaluation.node_evaluation.use_mean_node_loss:
        for nid, mean_loss in node2mean_loss.items():
            y_truth.append(node_labels[nid])
            y_pred.append(int(mean_loss > thr))
    
    # Thresholds each node based on if an edge loss involving this node is greater
    # than the threshold
    else:
        node_preds = {}
        edge_preds = []
        for (srcnode, dstnode), loss, edge_label in tqdm(zip(edge_index, losses, edge_labels), desc="Edge thresholding"):
            if loss > thr:
                node_preds[srcnode] = 1
            else:
                if srcnode not in node_preds:
                    node_preds[srcnode] = 0

            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                if loss > thr:
                    node_preds[dstnode] = 1
                else:
                    if dstnode not in node_preds:
                        node_preds[dstnode] = 0
            
            edge_preds.append(int(loss > thr))

        for nid in node_labels:
            y_truth.append(node_labels[nid])
            y_pred.append(node_preds[nid])

        # logger.info("\nEdge detection")
        # classifier_evaluation(edge_labels, edge_preds, losses)
    
    logger.info("\nNode detection")
    return classifier_evaluation(y_truth, y_pred, node_mean)

def plot_precision_recall(y_true, y_scores, out_file):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    precision_ticks = [i / 20 for i in range(7)]  # Generates [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    plt.yticks(precision_ticks)

    plt.savefig(out_file)

def main(cfg):
    logger = get_logger(
        name="node_evaluation",
        filename=os.path.join(cfg.detection.evaluation._logs_dir, "node_evaluation.log"))

    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    best_precision, best_stats = 0.0, {}
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        print(f"\nEvaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = node_evaluation_without_triage(val_tw_path, test_tw_path, model_epoch_dir, logger, cfg)
            
        stats["epoch"] = int(re.findall(r'[+-]?\d*\.?\d+', model_epoch_dir)[0])
        stats["precision_recall_img"] = wandb.Image(os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png"))
        
        wandb.log(stats)
        
        if stats["precision"] > best_precision:
            best_precision = stats["precision"]
            best_stats = stats
        
    wandb.log(best_stats)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
