import os
from collections import defaultdict

import pandas as pd
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_discrimination_score,
    compute_discrimination_tp,
    get_detected_tps,
    get_metrics_if_all_attacks_detected,
    get_threshold,
    plot_detected_attacks_vs_precision,
    plot_discrimination_metric,
    plot_scores_neat,
    plot_scores_with_paths_edge_level,
    transform_attack2nodes_to_node2attacks,
)
from pidsmaker.utils.dataset_utils import get_rel2id_considering_triplets
from pidsmaker.utils.labelling import (
    get_attack_to_mal_edges,
    get_ground_truth_edges,
    ground_truth_attacks,
    has_event_labels,
)
from pidsmaker.utils.loss_files import compact_int_columns, read_losses
from pidsmaker.utils.utils import listdir_sorted, log, log_tqdm


def get_edge_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    # With event labels, edges carry their labels in the loss files (`y`)
    ground_truth_edges = set() if has_event_labels(cfg) else get_ground_truth_edges(cfg)
    threshold_method = cfg.evaluation.edge_evaluation.threshold_method
    alpha = getattr(cfg.evaluation.edge_evaluation, "alpha", 1.0) or 1.0
    
    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method, alpha)
    else:
        thr = get_threshold(val_tw_path, threshold_method, alpha)
    log(f"Threshold: {thr:.3f}")

    scores, y_true, y_hat, src_dst_t_type = [], [], [], []
    edge_type_map = get_rel2id_considering_triplets(cfg)
    filelist = listdir_sorted(test_tw_path)

    # Build edge records for saving (consistent with node_evaluation)
    all_edge_records = []

    for file in log_tqdm(filelist, desc="Compute edge labels"):
        filepath = os.path.join(test_tw_path, file)
        df = read_losses(filepath).to_dict(orient="records")

        for line in df:
            srcnode = str(line["srcnode"])
            dstnode = str(line["dstnode"])
            loss = line["loss"]
            t = line["time"]
            edge_type = edge_type_map[line["edge_type"]]
            edge = (srcnode, dstnode, t, edge_type)

            if "y" in line:
                # Event-level labels carried by the edges (datasets with labelled events)
                is_malicious = int(line["y"] > 0)
            else:
                # CSV-based labels use 3-tuples (no timestamp), DB-based use 4-tuples
                is_malicious = int(
                    edge in ground_truth_edges
                    or (srcnode, dstnode, edge_type) in ground_truth_edges
                )
            scores.append(loss)
            y_true.append(is_malicious)
            y_hat.append(int(loss > thr))
            src_dst_t_type.append(edge)

            all_edge_records.append({
                "srcnode": srcnode,
                "dstnode": dstnode,
                "loss": loss,
                "time": t,
                "edge_type": line["edge_type"],
                "src_is_malicious": is_malicious,
                "dst_is_malicious": is_malicious,
            })

    edge_records = pd.DataFrame(all_edge_records) if all_edge_records else None
    return scores, y_true, y_hat, src_dst_t_type, thr, edge_records


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    scores, y_truth, y_preds, src_dst_t_type, thr, edge_records = get_edge_predictions(
        val_tw_path, test_tw_path, cfg, **kwargs
    )
    if has_event_labels(cfg):
        # Edges carry their labels (`y`): malicious edges are assigned to the attack whose time
        # window contains them. This also holds when edges are merged (e.g. `merge_edges`), whose
        # times and types may differ from those of the labelled events of the database.
        attacks = ground_truth_attacks(cfg)
        attack_to_mal_edges = {i: set() for i in range(len(attacks))}
        for edge, y in zip(src_dst_t_type, y_truth):
            if y:
                t = int(edge[2])
                i = next((i for i, (_, s, e) in enumerate(attacks) if s <= t <= e), 0)
                attack_to_mal_edges[i].add(edge)
        log(f"Found {sum(y_truth)} malicious edges in the test set")
    else:
        attack_to_mal_edges = get_attack_to_mal_edges(cfg)
        log(
            f"Found {sum(y_truth)} / {sum(len(edges) for edges in attack_to_mal_edges.values())} malicious edges"
        )

    edge2attack = transform_attack2nodes_to_node2attacks(attack_to_mal_edges)

    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save edge-level scores (consistent with node_evaluation)
    if edge_records is not None:
        edge_scores_file = os.path.join(out_dir, f"edge_scores_{model_epoch_dir}.pkl")
        torch.save(compact_int_columns(edge_records), edge_scores_file)
        log(f"Saved {len(edge_records)} edge scores to {edge_scores_file}")

    adp_img_file = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    neat_scores_img_file = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")

    save_figures = cfg.evaluation.save_figures is not False  # unset: saved
    if save_figures:
        log(f"Saving figures to {out_dir}...")
    adp_score = plot_detected_attacks_vs_precision(
        scores, src_dst_t_type, edge2attack, y_truth, adp_img_file if save_figures else None
    )
    discrim_scores = compute_discrimination_score(scores, src_dst_t_type, edge2attack, y_truth)
    discrim_tp = compute_discrimination_tp(scores, src_dst_t_type, edge2attack, y_truth)
    if save_figures:
        plot_discrimination_metric(scores, y_truth, discrim_img_file)
        plot_scores_with_paths_edge_level(
            scores,
            y_truth,
            src_dst_t_type,
            tw_to_malicious_nodes,
            edge2attack,
            scores_img_file,
            cfg,
            thr,
        )
        plot_scores_neat(scores, y_truth, src_dst_t_type, edge2attack, neat_scores_img_file, thr)
    stats = classifier_evaluation(y_truth, y_preds, scores)

    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(
        scores, src_dst_t_type, attack_to_mal_edges
    )
    stats["fps_if_all_attacks_detected"] = fps
    stats["tps_if_all_attacks_detected"] = tps
    stats["precision_if_all_attacks_detected"] = precision
    stats["recall_if_all_attacks_detected"] = recall

    stats["adp_score"] = round(adp_score, 3)
    stats["threshold"] = thr

    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)

    attack2tps = get_detected_tps(scores, src_dst_t_type, edge2attack, y_truth, cfg)
    for attack, detected_tps in attack2tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)

    stats = {**stats, **discrim_tp}

    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
    torch.save(
        {
            "pred_scores": scores,
            "y_preds": y_preds,
            "y_truth": y_truth,
            "edges": src_dst_t_type,
            "edge2attack": edge2attack,
        },
        scores_file,
    )

    stats["scores_file"] = scores_file
    if save_figures:
        stats["neat_scores_img_file"] = neat_scores_img_file

    return stats
