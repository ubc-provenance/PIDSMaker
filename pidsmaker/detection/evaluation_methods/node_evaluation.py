import os
from collections import defaultdict

import pandas as pd
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_discrimination_score,
    compute_discrimination_tp,
    compute_kmeans_labels,
    datetime_to_ns_time_US_handle_nano,
    get_detected_tps_node_level,
    get_ground_truth_nids,
    get_metrics_if_all_attacks_detected,
    get_threshold,
    plot_detected_attacks_vs_precision,
    plot_discrimination_metric,
    plot_score_seen,
    plot_scores_neat,
    plot_scores_with_paths_node_level,
    reduce_losses_to_score,
    transform_attack2nodes_to_node2attacks,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack
from pidsmaker.utils.utils import (
    get_all_files_from_folders,
    get_node_to_path_and_type,
    listdir_sorted,
    log,
    log_tqdm,
)


def get_node_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")

    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method
    if threshold_method == "magic":  # data leaking by using test data
        thr = get_threshold(test_tw_path, threshold_method)
    else:
        thr = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_losses = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(log_tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient="records")
        for line in df:
            srcnode = line["srcnode"]
            dstnode = line["dstnode"]
            loss = line["loss"]

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

    # For plotting the scores of seen and unseen nodes
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())

    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_losses.items():
        pred_score = reduce_losses_to_score(
            losses, cfg.detection.evaluation.node_evaluation.threshold_method
        )

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        if use_kmeans:  # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            results[node_id]["y_hat"] = int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr


def get_node_predictions_node_level(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")

    threshold_method = cfg.detection.evaluation.node_evaluation.threshold_method
    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method)
    else:
        thr = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_values = defaultdict(lambda: defaultdict(list))
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(log_tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient="records")
        for line in df:
            node = line["node"]
            loss = line["loss"]

            node_to_values[node]["loss"].append(loss)
            node_to_values[node]["tw"].append(tw)

            if "threatrace_score" in line:
                node_to_values[node]["threatrace_score"].append(line["threatrace_score"])
            if "correct_pred" in line:
                node_to_values[node]["correct_pred"].append(line["correct_pred"])
            if "flash_score" in line:
                node_to_values[node]["flash_score"].append(line["flash_score"])
            if "magic_score" in line:
                node_to_values[node]["magic_score"].append(line["magic_score"])

            if loss > node_to_max_loss[node]:
                node_to_max_loss[node] = loss
                node_to_max_loss_tw[node] = tw

    # For plotting the scores of seen and unseen nodes
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())

    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_values.items():
        threatrace_label = 0
        flash_label = 0
        detected_tw = None
        if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
            max_score = 0
            pred_score = max(losses["threatrace_score"])

            for score, node_type_pred, tw in zip(
                losses["threatrace_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    threatrace_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
            max_score = 0
            pred_score = max(losses["flash_score"])

            for score, node_type_pred, tw in zip(
                losses["flash_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.detection.evaluation.node_evaluation.threshold_method == "magic":
            max_score = 0
            pred_score = max(losses["magic_score"])

            for score, tw in zip(losses["magic_score"], losses["tw"]):
                if score > thr and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw

        else:
            pred_score = reduce_losses_to_score(
                losses["loss"],
                cfg.detection.evaluation.node_evaluation.threshold_method,
            )

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        # We need the detected TW range to check if the detected node spans in an attack TW
        detected_tw = detected_tw or node_to_max_loss_tw.get(node_id, None)
        if detected_tw is not None:
            results[node_id]["time_range"] = [
                datetime_to_ns_time_US_handle_nano(tw) for tw in filelist[detected_tw].split("~")
            ]
        else:
            results[node_id]["time_range"] = None

        if use_kmeans:  # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            if cfg.detection.evaluation.node_evaluation.threshold_method == "threatrace":
                results[node_id]["y_hat"] = threatrace_label
            elif cfg.detection.evaluation.node_evaluation.threshold_method == "flash":
                results[node_id]["y_hat"] = flash_label
            else:
                results[node_id]["y_hat"] = int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr


def analyze_false_positives(
    y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes
):
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_truth, y_preds)) if pred and not true]
    malicious_tws = set(tw_to_malicious_nodes.keys())
    num_fps_in_malicious_tw = 0

    for i in fp_indices:
        is_in_malicious_tw = max_val_loss_tw[i] in malicious_tws
        num_fps_in_malicious_tw += int(is_in_malicious_tw)

    fp_in_malicious_tw_ratio = (
        num_fps_in_malicious_tw / len(fp_indices) if len(fp_indices) > 0 else float("nan")
    )
    return fp_in_malicious_tw_ratio


def main(
    val_tw_path,
    test_tw_path,
    model_epoch_dir,
    cfg,
    tw_to_malicious_nodes,
    **kwargs,
):
    if cfg._is_node_level:
        get_preds_fn = get_node_predictions_node_level
    else:
        get_preds_fn = get_node_predictions

    results, thr = get_preds_fn(cfg=cfg, val_tw_path=val_tw_path, test_tw_path=test_tw_path)

    # save results for future checking
    os.makedirs(cfg.detection.evaluation._results_dir, exist_ok=True)
    results_save_dir = os.path.join(cfg.detection.evaluation._results_dir, "results.pth")
    torch.save(results, results_save_dir)
    log(f"Resutls saved to {results_save_dir}")

    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.detection.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    # pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    adp_img_file = os.path.join(
        out_dir, f"adp_curve_{model_epoch_dir}.png"
    )  # average detection precision
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    # simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    neat_scores_img_file = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")
    seen_score_img_file = os.path.join(out_dir, f"seen_score_{model_epoch_dir}.png")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")

    attack_to_GPs = get_GP_of_each_attack(cfg)
    attack_to_TPs = defaultdict(int)

    log("Analysis of malicious nodes:")
    nodes, y_truth, y_preds, pred_scores, max_val_loss_tw = [], [], [], [], []
    is_seen = []
    for nid, result in results.items():
        nodes.append(nid)
        score, y_hat, y_true, max_tw = (
            result["score"],
            result["y_hat"],
            result["y_true"],
            result["tw_with_max_loss"],
        )
        seen_flag = result["is_seen"]
        is_seen.append(seen_flag)
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)
        max_val_loss_tw.append(max_tw)

        if y_true == 1:
            log(
                f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:"
                + (" ✅ " if y_true == y_hat else " ❌ ")
                + (node_to_path[nid]["path"])
            )

            if y_hat:
                for att, d in attack_to_GPs.items():
                    if nid in d["nids"]:
                        attack_to_TPs[att] += 1

    attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
    node2attacks = transform_attack2nodes_to_node2attacks(attack2nodes)

    # Plots the PR curve and scores for mean node loss
    log(f"Saving figures to {out_dir}...")
    # plot_precision_recall(pred_scores, y_truth, pr_img_file)
    adp_score = plot_detected_attacks_vs_precision(
        pred_scores, nodes, node2attacks, y_truth, adp_img_file
    )
    discrim_scores = compute_discrimination_score(pred_scores, nodes, node2attacks, y_truth)
    plot_discrimination_metric(pred_scores, y_truth, discrim_img_file)
    discrim_tp = compute_discrimination_tp(pred_scores, nodes, node2attacks, y_truth)
    # plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)
    plot_scores_with_paths_node_level(
        pred_scores,
        y_truth,
        nodes,
        max_val_loss_tw,
        tw_to_malicious_nodes,
        node2attacks,
        scores_img_file,
        cfg,
        thr,
    )
    plot_scores_neat(pred_scores, y_truth, nodes, node2attacks, neat_scores_img_file, thr)
    # plot_score_seen(pred_scores, is_seen, seen_score_img_file)
    stats = classifier_evaluation(y_truth, y_preds, pred_scores)

    fp_in_malicious_tw_ratio = analyze_false_positives(
        y_truth,
        y_preds,
        pred_scores,
        max_val_loss_tw,
        nodes,
        tw_to_malicious_nodes,
    )
    stats["fp_in_malicious_tw_ratio"] = round(fp_in_malicious_tw_ratio, 3)

    log("TPs per attack:")
    tps_in_atts = []
    for att, tps in attack_to_TPs.items():
        log(f"attack {att}: {tps}")
        tps_in_atts.append((att, tps))

    stats["percent_detected_attacks"] = (
        round(len(attack_to_TPs) / len(attack_to_GPs), 2) if len(attack_to_GPs) > 0 else 0
    )

    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(
        pred_scores, nodes, attack_to_GPs
    )
    stats["fps_if_all_attacks_detected"] = fps
    stats["tps_if_all_attacks_detected"] = tps
    stats["precision_if_all_attacks_detected"] = precision
    stats["recall_if_all_attacks_detected"] = recall

    stats["adp_score"] = round(adp_score, 3)

    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)

    attack2tps = get_detected_tps_node_level(pred_scores, nodes, node2attacks, y_truth, cfg)
    for attack, detected_tps in attack2tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)

    stats = {**stats, **discrim_tp}

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")

    torch.save(results, results_file)
    torch.save(stats, stats_file)

    torch.save(
        {
            "pred_scores": pred_scores,
            "y_preds": y_preds,
            "y_truth": y_truth,
            "nodes": nodes,
            "node2attacks": node2attacks,
        },
        scores_file,
    )

    stats["scores_file"] = scores_file
    stats["neat_scores_img_file"] = neat_scores_img_file

    return stats
