from collections import defaultdict

import torch
import numpy as np
import wandb

from provnet_utils import *
from config import *
from .evaluation_utils import *
from labelling import get_GP_of_each_attack


def get_node_predictions(val_tw_path, test_tw_path, cfg):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.node_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_losses = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)
    
    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        for line in df:
            srcnode = line['srcnode']
            dstnode = line['dstnode']
            loss = line['loss']
            
            # Scores
            node_to_losses[srcnode].append(loss)
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                node_to_losses[dstnode].append(loss)
                
            # If max-val thr is used, we want to keep track when the node with max loss happens
            if loss > node_to_max_loss[srcnode]:
                node_to_max_loss[srcnode] = loss
                node_to_max_loss_tw[srcnode] = tw
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                if loss > node_to_max_loss[dstnode]:
                    node_to_max_loss[dstnode] = loss
                    node_to_max_loss_tw[dstnode] = tw
                    
    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_losses.items():
        pred_score = reduce_losses_to_score(losses, cfg.detection.evaluation.node_evaluation.threshold_method)

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        
        if use_kmeans: # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            results[node_id]["y_hat"] = int(pred_score > thr)
        
    if use_kmeans:
        results = compute_kmeans_labels(results, topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K)
        
    return results

def get_node_predictions_node_level(val_tw_path, test_tw_path, cfg):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.node_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_values = defaultdict(lambda: defaultdict(list))
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)
    
    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        for line in df:
            node = line['node']
            loss = line['loss']
            
            node_to_values[node]["loss"].append(loss)
            node_to_values[node]["tw"].append(tw)
            
            if "threatrace_score" in line:
                node_to_values[node]["threatrace_score"].append(line["threatrace_score"])
            if "correct_pred" in line:
                node_to_values[node]["correct_pred"].append(line["correct_pred"])
            if "flash_score" in line:
                node_to_values[node]["flash_score"].append(line["flash_score"])

            if loss > node_to_max_loss[node]:
                node_to_max_loss[node] = loss
                node_to_max_loss_tw[node] = tw
                    
    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_values.items():
        threatrace_label = 0
        flash_label = 0
        detected_tw = None
        if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
            max_score = 0
            pred_score = max(losses["threatrace_score"])
        
            for score, node_type_pred, tw in zip(losses["threatrace_score"], losses["correct_pred"], losses["tw"]):
                if score > thr and node_type_pred and score > max_score:
                    threatrace_label = 1
                    max_score = score
                    detected_tw = tw
        
        elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
            max_score = 0
            pred_score = max(losses["flash_score"])

            for score, node_type_pred, tw in zip(losses["flash_score"], losses["correct_pred"], losses["tw"]):
                if score > thr and node_type_pred and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw
        
        else:
            pred_score = reduce_losses_to_score(losses["loss"], cfg.detection.evaluation.node_evaluation.threshold_method)
        
        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        
        # We need the detected TW range to check if the detected node spans in an attack TW
        detected_tw = detected_tw or node_to_max_loss_tw.get(node_id, None)
        if detected_tw is not None:
            results[node_id]["time_range"] = [datetime_to_ns_time_US(tw) for tw in filelist[detected_tw].split("~")]
        else:
            results[node_id]["time_range"] = None
        
        if use_kmeans: # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
                results[node_id]["y_hat"] = threatrace_label
            elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
                results[node_id]["y_hat"] = flash_label
            else:
                results[node_id]["y_hat"] = int(pred_score > thr)
        
    if use_kmeans:
        results = compute_kmeans_labels(results, topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K)
        
    return results


def analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes):
    log(f"Analysis of false positives:")
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_truth, y_preds)) if pred and not true]
    malicious_tws = set(tw_to_malicious_nodes.keys())
    num_fps_in_malicious_tw = 0
    
    for i in fp_indices:
        is_in_malicious_tw = max_val_loss_tw[i] in malicious_tws
        num_fps_in_malicious_tw += int(is_in_malicious_tw)

        # log(f"FP node {nodes[i]} -> max loss: {pred_scores[i]:.3f} | max TW: {max_val_loss_tw[i]} "
        #     f"| is malicious TW: " + (" ✅" if is_in_malicious_tw else " ❌"))
    
    fp_in_malicious_tw_ratio = num_fps_in_malicious_tw / len(fp_indices) if len(fp_indices) > 0 else float("nan")
    log(f"Percentage of FPs present in malicious TWs: {fp_in_malicious_tw_ratio:.3f}")
    return fp_in_malicious_tw_ratio

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    get_preds_fn = get_node_predictions_node_level if cfg._is_node_level else get_node_predictions
    results = get_preds_fn(val_tw_path, test_tw_path, cfg)
    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    dor_img_file = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
    
    attack_to_GPs = get_GP_of_each_attack(cfg)
    attack_to_TPs = defaultdict(int)

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
            log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (node_to_path[nid]['path']))
            
            if y_hat:
                for att, d in attack_to_GPs.items():
                    start_att, end_att = d["time_range"]
                    start_node, end_node = result["time_range"]
                    if nid in d["nids"] and (start_node <= start_att <= end_node or start_node <= end_att <= end_node):
                        attack_to_TPs[att] += 1

    # Plots the PR curve and scores for mean node loss
    print(f"Saving figures to {out_dir}...")
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_dor_recall_curve(pred_scores, y_truth, dor_img_file)
    plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)
    plot_scores_with_paths(pred_scores, y_truth, nodes, max_val_loss_tw, tw_to_malicious_nodes, scores_img_file, cfg)
    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    
    fp_in_malicious_tw_ratio = analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes)
    stats["fp_in_malicious_tw_ratio"] = fp_in_malicious_tw_ratio
    
    tps_in_atts = []
    for att, tps in attack_to_TPs.items():
        log(f"attack {att}: {tps}")
        tps_in_atts.append((att, tps))

    stats['percent_detected_attacks'] = round(len(attack_to_GPs) / len(attack_to_TPs), 2) if len(attack_to_TPs) > 0 else 0
    
    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")

    torch.save(results, results_file)
    torch.save(stats, stats_file)
    
    return stats
