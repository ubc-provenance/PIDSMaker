"""Node-level anomaly detection evaluation.

Aggregates edge-level anomaly scores to node-level scores and evaluates
detection performance against ground truth. Supports multiple threshold methods
(magic, max-val, nodlink) and score reduction strategies (sum, mean, max, percentile).

Key metrics:
- Precision, Recall, F1 at node level
- ADP (Average Detection Precision)
- Discrimination score (attack nodes ranked before benign nodes)
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_discrimination_score,
    compute_discrimination_tp,
    compute_tp_per_attack_thr,
    compute_kmeans_labels,
    datetime_to_ns_time_US_handle_nano,
    get_detected_tps_node_level,
    get_ground_truth_nids,
    get_metrics_if_all_attacks_detected,
    get_threshold,
    get_threshold_per_type,
    plot_detected_attacks_vs_precision,
    plot_discrimination_metric,
    plot_score_seen,
    plot_scores_neat,
    plot_scores_with_paths_node_level,
    reduce_losses_to_score,
    transform_attack2nodes_to_node2attacks,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack
from pidsmaker.detection.evaluation_methods.plot import plot_scores_distribution
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    listdir_sorted,
    log,
    log_tqdm,
)
from pidsmaker.utils.dataset_utils import (
    get_node_to_path_and_type
)

# Cache for train_node_set — loading all train graphs from disk is expensive
# and the result is the same across epochs.
_train_node_set_cache = {}


def _get_train_node_set(cfg):
    cache_key = cfg.transformation._graphs_dir
    if cache_key not in _train_node_set_cache:
        train_set_paths = get_all_graphs_for_dates(cache_key, cfg.dataset.train_dates)
        train_node_set = set()
        for train_path in train_set_paths:
            train_graph = torch.load(train_path)
            train_node_set |= set(train_graph.nodes())
        _train_node_set_cache[cache_key] = train_node_set
    return _train_node_set_cache[cache_key]


def _vectorized_node_score(grouped_loss, threshold_method):
    """Compute per-node score using vectorized groupby instead of per-node reduce_losses_to_score."""
    method = threshold_method.strip()
    if method == "mean_val_loss":
        return grouped_loss.mean()
    elif method == "percentile":
        return grouped_loss.quantile(0.95)
    else:  # max_val_loss, threatrace, flash, nodlink, fixed_zero
        return grouped_loss.max()



def get_node_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    """Compute node-level anomaly scores from edge-level losses.

    Aggregates edge losses to node scores using configured reduction strategy,
    applies threshold, and generates predictions for evaluation.

    Args:
        val_tw_path: Path to validation time window results
        test_tw_path: Path to test time window results
        cfg: Configuration with evaluation settings
        **kwargs: Additional arguments

    Returns:
        dict: Contains metrics, predictions, scores, and visualization data
    """
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")


    threshold_method = cfg.evaluation.node_evaluation.threshold_method
    alpha = cfg.evaluation.node_evaluation.max_val_loss.alpha or 1.0
    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method, alpha)
    else:
        thr = get_threshold(val_tw_path, threshold_method, alpha)
    log(f"Threshold: {thr:.3f}")

    use_dst = cfg.evaluation.node_evaluation.use_dst_node_loss

    # Read all CSVs and tag with TW index
    filelist = listdir_sorted(test_tw_path)
    dfs = []
    for tw, fname in enumerate(log_tqdm(filelist, desc="Compute labels")):
        df = pd.read_csv(os.path.join(test_tw_path, fname))
        df["tw"] = tw
        dfs.append(df)

    if not dfs:
        return defaultdict(dict), thr, None

    all_df = pd.concat(dfs, ignore_index=True)

    # Build edge records DataFrame directly (vectorized isin instead of per-row `in`)
    all_edge_records = all_df[["srcnode", "dstnode", "loss", "tw"]].copy()
    all_edge_records["edge_type"] = all_df["edge_type"] if "edge_type" in all_df.columns else -1
    all_edge_records["time"] = all_df["time"] if "time" in all_df.columns else 0
    all_edge_records["src_is_malicious"] = all_df["srcnode"].isin(ground_truth_nids).astype(int)
    all_edge_records["dst_is_malicious"] = all_df["dstnode"].isin(ground_truth_nids).astype(int)

    # Combine src (and optionally dst) into a unified node-loss-tw table
    src_part = all_df[["srcnode", "loss", "tw"]].rename(columns={"srcnode": "node"})
    if use_dst:
        dst_part = all_df[["dstnode", "loss", "tw"]].rename(columns={"dstnode": "node"})
        node_loss_tw = pd.concat([src_part, dst_part], ignore_index=True)
    else:
        node_loss_tw = src_part

    # Compute per-node score via vectorized groupby
    grouped = node_loss_tw.groupby("node")["loss"]
    node_scores = _vectorized_node_score(grouped, threshold_method)

    # Compute TW of max loss per node (idxmax returns first occurrence, same as strict >)
    max_loss_idx = grouped.idxmax()
    max_rows = node_loss_tw.loc[max_loss_idx.values]
    node_to_max_loss_tw = dict(zip(max_rows["node"].values, max_rows["tw"].values))

    train_node_set = _get_train_node_set(cfg)

    use_kmeans = cfg.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, score in node_scores.items():
        score = float(score)
        results[node_id]["score"] = score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        if use_kmeans:
            results[node_id]["y_hat"] = 0
        else:
            results[node_id]["y_hat"] = int(score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr, all_edge_records


def get_node_predictions_node_level(val_tw_path, test_tw_path, cfg, **kwargs):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")

    threshold_method = cfg.evaluation.node_evaluation.threshold_method
    alpha = cfg.evaluation.node_evaluation.max_val_loss.alpha or 1.0
    thr_by_type = None
    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method, alpha)
    elif threshold_method == "ocrapt":
        thr_by_type = get_threshold_per_type(
            val_tw_path, cfg,
            min_contamination=cfg.evaluation.node_evaluation.ocrapt_min_contamination,
            max_contamination=cfg.evaluation.node_evaluation.ocrapt_contamination,
        )
        thr = float(np.mean(list(thr_by_type.values()))) if thr_by_type else 0.0
    else:
        thr = get_threshold(val_tw_path, threshold_method, alpha)
    log(f"Threshold: {thr:.3f}" + (f" (per-type: {thr_by_type})" if thr_by_type else ""))

    node_to_type = {nid: info["type"] for nid, info in get_node_to_path_and_type(cfg).items()} \
        if thr_by_type is not None else None

    node_to_values = defaultdict(lambda: defaultdict(list))
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(log_tqdm(filelist, desc="Compute labels")):
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

    train_node_set = _get_train_node_set(cfg)

    use_kmeans = cfg.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_values.items():
        threatrace_label = 0
        flash_label = 0
        detected_tw = None
        if cfg.evaluation.node_evaluation.threshold_method == "threatrace":
            max_score = 0
            pred_score = max(losses["threatrace_score"])

            for score, node_type_pred, tw in zip(
                losses["threatrace_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    threatrace_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.evaluation.node_evaluation.threshold_method == "flash":
            max_score = 0
            pred_score = max(losses["flash_score"])

            for score, node_type_pred, tw in zip(
                losses["flash_score"], losses["correct_pred"], losses["tw"]
            ):
                if score > thr and node_type_pred and score > max_score:
                    flash_label = 1
                    max_score = score
                    detected_tw = tw

        elif cfg.evaluation.node_evaluation.threshold_method == "magic":
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
                cfg.evaluation.node_evaluation.threshold_method,
            )

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        results[node_id]["is_seen"] = int(str(node_id) in train_node_set)

        # We need the detected TW range to check if the detected node spans in an attack TW
        detected_tw = detected_tw or node_to_max_loss_tw.get(node_id, None)
        if detected_tw is not None:
            results[node_id]["time_range"] = [
                datetime_to_ns_time_US_handle_nano(tw, timezone=cfg.dataset.timezone) for tw in filelist[detected_tw].split("~")
            ]
        else:
            results[node_id]["time_range"] = None

        if use_kmeans:  # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            if cfg.evaluation.node_evaluation.threshold_method == "threatrace":
                results[node_id]["y_hat"] = threatrace_label
            elif cfg.evaluation.node_evaluation.threshold_method == "flash":
                results[node_id]["y_hat"] = flash_label
            elif thr_by_type is not None:
                node_thr = thr_by_type.get(node_to_type.get(node_id), thr)
                results[node_id]["y_hat"] = int(pred_score > node_thr)
            else:
                results[node_id]["y_hat"] = int(pred_score > thr)

    if use_kmeans:
        results = compute_kmeans_labels(
            results,
            topk_K=cfg.evaluation.node_evaluation.kmeans_top_K,
        )
    return results, thr, None


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
    if cfg._is_node_level and not cfg._is_hybrid_loss:
        get_preds_fn = get_node_predictions_node_level
    else:
        get_preds_fn = get_node_predictions

    results, thr, edge_records = get_preds_fn(cfg=cfg, val_tw_path=val_tw_path, test_tw_path=test_tw_path)

    # save results for future checking
    # os.makedirs(cfg.evaluation._results_dir, exist_ok=True)
    # results_save_dir = os.path.join(cfg.evaluation._results_dir, "results.pth")
    # torch.save(results, results_save_dir)
    # results_epoch_save_dir = os.path.join(cfg.evaluation._results_dir, f"results_{model_epoch_dir}.pth")
    # torch.save(results, results_epoch_save_dir)
    # log(f"Resutls saved to {results_save_dir}")

    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save edge-level scores for fine-grained analysis
    if edge_records is not None:
        edge_scores_file = os.path.join(out_dir, f"edge_scores_{model_epoch_dir}.pkl")
        edge_df = edge_records if isinstance(edge_records, pd.DataFrame) else pd.DataFrame(edge_records)
        torch.save(edge_df, edge_scores_file)
        log(f"Saved {len(edge_df)} edge scores to {edge_scores_file}")
    # pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    adp_img_file = os.path.join(
        out_dir, f"adp_curve_{model_epoch_dir}.png"
    )  # average detection precision
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    # simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    neat_scores_img_file = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")
    # seen_score_img_file = os.path.join(out_dir, f"seen_score_{model_epoch_dir}.png")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
    distrib_img_file = os.path.join(out_dir, f"distrib_{model_epoch_dir}.png")

    attack_to_GPs = get_GP_of_each_attack(cfg)
    attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
    node2attacks = transform_attack2nodes_to_node2attacks(attack2nodes)
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

            if y_hat and nid in node2attacks:
                for att in node2attacks[nid]:
                    attack_to_TPs[att] += 1

    # Plots the PR curve and scores for mean node loss
    log(f"Saving figures to {out_dir}...")
    # plot_precision_recall(pred_scores, y_truth, pr_img_file)
    adp_score = plot_detected_attacks_vs_precision(
        pred_scores, nodes, node2attacks, y_truth, adp_img_file
    )
    discrim_scores = compute_discrimination_score(pred_scores, nodes, node2attacks, y_truth)
    plot_discrimination_metric(pred_scores, y_truth, discrim_img_file)
    discrim_tp = compute_discrimination_tp(pred_scores, nodes, node2attacks, y_truth)
    tp_per_attack = compute_tp_per_attack_thr(pred_scores, nodes, node2attacks, y_truth, thr)
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
    node_to_type = {nid: info["type"] for nid, info in node_to_path.items()}
    plot_scores_distribution(pred_scores, y_truth, nodes, node2attacks, distrib_img_file,
                             node_to_type=node_to_type)
    
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
    stats["threshold"] = thr

    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)

    attack2tps = get_detected_tps_node_level(pred_scores, nodes, node2attacks, y_truth, cfg)
    for attack, detected_tps in attack2tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)

    stats = {**stats, **discrim_tp, **tp_per_attack}

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    torch.save(results, results_file)
    torch.save(stats, stats_file)
    
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
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
    
    gnn_models_dir = cfg.training._trained_models_dir
    model_path = os.path.join(gnn_models_dir, model_epoch_dir)
    stats["model_path"] = model_path

    return stats
    