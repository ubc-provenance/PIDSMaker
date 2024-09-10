import torch
from data_utils import *
from provnet_utils import *

from labelling import get_ground_truth, get_GP_of_each_attack
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from .evaluation_utils import *

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
    all_nids = get_set_nodes(split_files=cfg.dataset.test_files,cfg=cfg)
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

def main(cfg, model, epoch):
    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = set(str(nid) for nid in GP_nids)

    attack_to_GPs = get_GP_of_each_attack(cfg)

    tw_to_malicious_nodes = compute_tw_labels(cfg)

    log("Processing testing results ")
    in_dir = cfg.detection.gnn_training._edge_losses_dir
    node_tw_list = listdir_sorted(in_dir)
    tw_to_data = torch.load(node_tw_list[epoch])

    results = {}
    nid_to_max_score = {}
    nid_to_max_score_tw = {}
    # note that currently there is only one epoch
    for tw, data in tw_to_data.items():
        for i in range(len(data['nids'])):
            node_id = data['nids'][i]
            score = data['score'][i]
            y_hat = data['y_hat'][i]
            y_true = 1 if node_id in GPs else 0

            if node_id not in results:
                results[node_id] = {}
                results[node_id]['y_true'] = 0
                results[node_id]['y_hat'] = 0
            results[node_id]['y_true'] = results[node_id]['y_true'] or y_true
            results[node_id]['y_hat'] = results[node_id]['y_hat'] or y_hat

            if node_id not in nid_to_max_score:
                nid_to_max_score[node_id] = score
                nid_to_max_score_tw[node_id] = tw

            if score > nid_to_max_score[node_id]:
                nid_to_max_score[node_id] = score
                nid_to_max_score_tw[node_id] = tw

    for n in results.keys():
        results[n]['score'] = nid_to_max_score[n]
        results[n]['tw_with_max_loss'] = nid_to_max_score_tw[n]

    results = uniforming_nodes(results, cfg)

    node_to_path = get_node_to_path_and_type(cfg)
    model_epoch_dir = "flash_evaluation"

    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    dor_img_file = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")

    log("Analysis of malicious nodes:")
    attack_to_TPs = {}
    for attack in attack_to_GPs.keys():
        attack_to_TPs[attack] = 0
    nodes, y_truth, y_preds, pred_scores, max_val_loss_tw = [], [], [], [], []
    for nid, result in results.items():
        nodes.append(int(nid))
        score, y_hat, y_true, max_tw = result["score"], result["y_hat"], result["y_true"], result["tw_with_max_loss"]
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)
        max_val_loss_tw.append(max_tw)

        if (y_hat == 1) and (y_true == 1):
            for att, gps in attack_to_GPs.items():
                if int(nid) in gps:
                    attack_to_TPs[att] += 1

        if y_true == 1:
            log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (
                node_to_path[int(nid)]['path']))

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
    stats["simple_scores_img"] = wandb.Image(os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png"))
    stats["dor_img"] = wandb.Image(os.path.join(out_dir, f"dor_{model_epoch_dir}.png"))

    for k,v in stats.items():
        log(k, " : ", v)

    log("Detected malicious nodes in each attacks:")
    tps_in_atts = []
    for att, tps in attack_to_TPs.items():
        log(f"attack {att}: {tps}")
        tps_in_atts.append((att, tps))

    detected_attacks = {
        'tps_in_atts': tps_in_atts,
    }

    wandb.log(detected_attacks)

    wandb.log(stats)

    best_stats = stats
    wandb.log(best_stats)

    return stats


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)