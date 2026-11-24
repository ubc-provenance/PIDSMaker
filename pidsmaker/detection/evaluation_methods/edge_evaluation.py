"""Edge-level evaluation: each test edge is scored by its loss.

Test sets of network datasets have tens of millions of edges, so the predictions are kept as
numpy arrays, and the edges are given to the metrics as a lazy sequence of (src, dst, time,
edge_type) tuples (`EdgeTuples`): tuples are only built for the edges the metrics need
(malicious edges, top-ranked edges, ties).
"""

import os
from collections.abc import Sequence

import numpy as np
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


class EdgeTuples(Sequence):
    """The (src, dst, time, edge_type) tuples of the test edges, the edge identifiers of the
    ground truth (node IDs as strings, edge type as its name), built on access."""

    def __init__(self, src, dst, time, edge_type, type_names):
        self.src, self.dst, self.time, self.edge_type = src, dst, time, edge_type
        self.type_names = type_names

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.take(np.arange(len(self))[i])
        return self.take([i])[0]

    def take(self, idx):
        """Tuples of the edges at the positions `idx`."""
        idx = np.asarray(idx, dtype=np.int64)
        names = self.type_names
        return [
            (str(s), str(d), t, names[e])
            for s, d, t, e in zip(
                self.src[idx].tolist(),
                self.dst[idx].tolist(),
                self.time[idx].tolist(),
                self.edge_type[idx].tolist(),
            )
        ]

    def candidates(self, edge_tuples):
        """Positions of the edges whose source is the source of one of `edge_tuples`: a superset
        of the edges equal to one of them (sources are compared as integers, without building
        the tuples of all the edges)."""
        srcs = set()
        for edge in edge_tuples:
            try:
                srcs.add(int(edge[0]))
            except (TypeError, ValueError):
                pass  # not the string of an integer: cannot be the source of an edge
        return np.flatnonzero(np.isin(self.src, np.array(sorted(srcs), dtype=self.src.dtype)))

    def attack_hits(self, edge2attack):
        """Same as `evaluation_utils.attack_hits`, checking only the candidate edges."""
        idx = self.candidates(edge2attack.keys())
        positions, attacks = [], []
        for i, edge in zip(idx.tolist(), self.take(idx)):
            found = edge2attack.get(edge)
            if found is not None:
                positions.append(i)
                attacks.append(found)
        return np.array(positions, dtype=np.int64), attacks


def get_edge_predictions(val_tw_path, test_tw_path, cfg, **kwargs):
    """(scores, y_true, y_hat, edges, threshold, loss columns) of the test edges: arrays, and
    the edges as `EdgeTuples`."""
    threshold_method = cfg.evaluation.edge_evaluation.threshold_method
    alpha = getattr(cfg.evaluation.edge_evaluation, "alpha", 1.0) or 1.0

    if threshold_method == "magic":
        thr = get_threshold(test_tw_path, threshold_method, alpha)
    else:
        thr = get_threshold(val_tw_path, threshold_method, alpha)
    log(f"Threshold: {thr:.3f}")

    filelist = listdir_sorted(test_tw_path)
    dfs = [
        read_losses(os.path.join(test_tw_path, file))
        for file in log_tqdm(filelist, desc="Compute edge labels")
    ]
    columns = ("srcnode", "dstnode", "loss", "time", "edge_type")
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame({c: [] for c in columns})

    edge_type = df["edge_type"].to_numpy()
    type_map = get_rel2id_considering_triplets(cfg)
    edges = EdgeTuples(
        df["srcnode"].to_numpy(),
        df["dstnode"].to_numpy(),
        df["time"].to_numpy(),
        edge_type,
        {t: type_map[t] for t in np.unique(edge_type).tolist()},
    )
    scores = df["loss"].to_numpy(dtype=np.float64)
    y_hat = (scores > thr).astype(np.int8)

    if "y" in df.columns:
        # Event-level labels carried by the edges (datasets with labelled events)
        y_true = (df["y"].to_numpy() > 0).astype(np.int8)
    else:
        y_true = np.zeros(len(df), dtype=np.int8)
        ground_truth_edges = get_ground_truth_edges(cfg)
        # CSV-based labels use 3-tuples (no timestamp), DB-based use 4-tuples
        idx = edges.candidates(ground_truth_edges)
        for i, edge in zip(idx.tolist(), edges.take(idx)):
            if edge in ground_truth_edges or (edge[0], edge[1], edge[3]) in ground_truth_edges:
                y_true[i] = 1
    return scores, y_true, y_hat, edges, thr, df


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    scores, y_truth, y_preds, edges, thr, losses = get_edge_predictions(
        val_tw_path, test_tw_path, cfg, **kwargs
    )
    if has_event_labels(cfg):
        # Edges carry their labels (`y`): malicious edges are assigned to the attack whose time
        # window contains them. This also holds when edges are merged (e.g. `merge_edges`), whose
        # times and types may differ from those of the labelled events of the database.
        attacks = ground_truth_attacks(cfg)
        attack_to_mal_edges = {i: set() for i in range(len(attacks))}
        for edge in edges.take(np.flatnonzero(y_truth)):
            t = int(edge[2])
            i = next((i for i, (_, s, e) in enumerate(attacks) if s <= t <= e), 0)
            attack_to_mal_edges[i].add(edge)
        log(f"Found {int(y_truth.sum())} malicious edges in the test set")
    else:
        attack_to_mal_edges = get_attack_to_mal_edges(cfg)
        log(
            f"Found {int(y_truth.sum())} / {sum(len(edges) for edges in attack_to_mal_edges.values())} malicious edges"
        )

    edge2attack = transform_attack2nodes_to_node2attacks(attack_to_mal_edges)

    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save edge-level scores (consistent with node_evaluation)
    edge_records = pd.DataFrame(
        {
            "srcnode": edges.src,
            "dstnode": edges.dst,
            "loss": scores,
            "time": edges.time,
            "edge_type": edges.edge_type,
            "src_is_malicious": y_truth.astype(np.int64),
            "dst_is_malicious": y_truth.astype(np.int64),
        }
    )
    edge_scores_file = os.path.join(out_dir, f"edge_scores_{model_epoch_dir}.pkl")
    torch.save(compact_int_columns(edge_records), edge_scores_file)
    log(f"Saved {len(edge_records)} edge scores to {edge_scores_file}")
    del edge_records, losses

    adp_img_file = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    neat_scores_img_file = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")
    discrim_img_file = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")

    save_figures = cfg.evaluation.save_figures is not False  # unset: saved
    if save_figures:
        log(f"Saving figures to {out_dir}...")
    adp_score = plot_detected_attacks_vs_precision(
        scores, edges, edge2attack, y_truth, adp_img_file if save_figures else None
    )
    discrim_scores = compute_discrimination_score(scores, edges, edge2attack, y_truth)
    discrim_tp = compute_discrimination_tp(scores, edges, edge2attack, y_truth)
    if save_figures:
        # The plots iterate over all the edges
        scores_list, y_list, edges_list = scores.tolist(), y_truth.tolist(), list(edges)
        plot_discrimination_metric(scores_list, y_list, discrim_img_file)
        plot_scores_with_paths_edge_level(
            scores_list,
            y_list,
            edges_list,
            tw_to_malicious_nodes,
            edge2attack,
            scores_img_file,
            cfg,
            thr,
        )
        plot_scores_neat(scores_list, y_list, edges_list, edge2attack, neat_scores_img_file, thr)
        del scores_list, y_list, edges_list
    stats = classifier_evaluation(y_truth, y_preds, scores)

    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(
        scores, edges, attack_to_mal_edges
    )
    stats["fps_if_all_attacks_detected"] = fps
    stats["tps_if_all_attacks_detected"] = tps
    stats["precision_if_all_attacks_detected"] = precision
    stats["recall_if_all_attacks_detected"] = recall

    stats["adp_score"] = round(adp_score, 3)
    stats["threshold"] = thr

    for k, v in discrim_scores.items():
        stats[k] = round(v, 4)

    attack2tps = get_detected_tps(scores, edges, edge2attack, y_truth, cfg)
    for attack, detected_tps in attack2tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)

    stats = {**stats, **discrim_tp}

    # Edges as a structured array (src, dst, time, edge_type ID), indexable like tuples; the
    # names of the edge type IDs are in `edge_type_names`
    edge_array = np.empty(
        len(edges),
        dtype=[("src", np.int64), ("dst", np.int64), ("time", np.int64), ("edge_type", np.int64)],
    )
    edge_array["src"], edge_array["dst"] = edges.src, edges.dst
    edge_array["time"], edge_array["edge_type"] = edges.time, edges.edge_type
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
    torch.save(
        {
            "pred_scores": scores,
            "y_preds": y_preds,
            "y_truth": y_truth,
            "edges": edge_array,
            "edge_type_names": edges.type_names,
            "edge2attack": edge2attack,
        },
        scores_file,
    )

    stats["scores_file"] = scores_file
    if save_figures:
        stats["neat_scores_img_file"] = neat_scores_img_file

    return stats
