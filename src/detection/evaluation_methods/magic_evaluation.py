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
from tqdm import tqdm

def transfer_result_to_node_evaluation(results, node_to_max_loss_tw):
    node_results = {}
    nid_to_max_score = {}
    nid_to_max_score_tw = {}

    for tw, nid2data in tqdm(results.items(),desc="Transferring results"):
        # node_results[tw] = {}

        for node_id, data in nid2data.items():
            if node_id not in node_results:
                node_results[node_id] = {}
                node_results[node_id]['y_true'] = 0
                node_results[node_id]['y_hat'] = 0

            y_true = data['y_true']
            y_hat = data['y_hat']
            score = data['score']
            try:
                node_results[node_id]['y_true'] = node_results[node_id]['y_true'] or y_true
                node_results[node_id]['y_hat'] = node_results[node_id]['y_hat'] or y_hat
            except KeyError:
                log(f"key error in tw: {tw} and node_id: {node_id}")
                log(f"data is {data}")
                log(f"node_results: {node_results[node_id]}")

            if node_id not in nid_to_max_score:
                nid_to_max_score[node_id] = score
                nid_to_max_score_tw[node_id] = tw

            if score > nid_to_max_score[node_id]:
                nid_to_max_score[node_id] = score
                nid_to_max_score_tw[node_id] = tw

    # for node_id in node_results.keys():
    #     node_results[node_id]['tw_with_max_loss'] = node_to_max_loss_tw[node_id]
    #     node_results[node_id]['score'] = results[node_to_max_loss_tw[node_id]][node_id]['score']

    for n in node_results.keys():
        node_results[n]['score'] = nid_to_max_score[n]
        node_results[n]['tw_with_max_loss'] = nid_to_max_score_tw[n]

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

def get_set_nodes(split_files, cfg):
    all_nids = set()
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    for graph_path in tqdm(sorted_paths, desc='Computing node number'):
        graph = torch.load(graph_path)
        all_nids |= set(graph.nodes())

    return all_nids

def uniforming_nodes(results, cfg):
    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = set(str(nid) for nid in GP_nids)
    log(f"There are {len(GPs)} GPs")

    log("Get testing nodes")
    all_nids = get_set_nodes(split_files=cfg.dataset.test_files, cfg=cfg)
    log(f'There are {len(all_nids)} testing set nodes')

    log("Generate results for testing set nodes")
    new_results = {}
    missing_num = 0

    if isinstance(list(results.keys())[0], int):
        key_type = 'int'
    elif isinstance(list(results.keys())[0], str):
        key_type = 'str'

    for n in all_nids:
        if key_type == 'int':
            node_id = int(n)
        elif key_type == 'str':
            node_id = str(n)

        if node_id in results.keys():
            new_results[node_id] = results[node_id]
        else:
            new_results[node_id] = {
                'score': 0,
                'tw_with_max_loss': 0,
                'y_hat': 0,
                'y_true': int(str(node_id) in GPs)
            }
            missing_num += 1
    log(f"There are {missing_num} missing nodes")

    return new_results

def main(cfg):
    tw_to_malicious_nodes = compute_tw_labels_for_magic(cfg)

    node_tw_results, node_to_max_loss_tw = magic_eval.get_node_predictions(cfg, tw_to_malicious_nodes)

    method = cfg.detection.evaluation.used_method.strip()
    if method == "magic_node_evaluation":
        results = transfer_result_to_node_evaluation(node_tw_results, node_to_max_loss_tw)
    else:
        log(f"Method {method} not supported.")

    results = uniforming_nodes(results, cfg)

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