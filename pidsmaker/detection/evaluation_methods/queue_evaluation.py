"""Queue-based evaluation for PIDS systems (Kairos-style).

This module implements queue-based attack investigation where anomalous edges
are grouped into queues using IDF (Inverse Document Frequency) scoring of nodes.
Supports backward/forward tracing from Points of Interest (POIs) to identify
attack propagation paths.

Key functions:
- cal_idf_kairos: Calculate IDF scores for nodes across time windows
- cal_anomaly_loss_kairos: Identify anomalous edges using loss thresholds
- is_include_key_word: Filter out benign system nodes
- cal_set_rel: Compute queue relationships based on shared high-IDF nodes
"""

import copy
import math
import os
from collections import defaultdict

import torch
import wandb

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_tw_labels,
    detection_mark,
    epoch_name,
    load_losses_per_time_window,
    log_best_epoch,
    log_epoch_summary,
)
from pidsmaker.utils import console
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_indexid2msg,
    listdir_sorted,
    log,
    log_tqdm,
    mean,
    std,
)


def cal_idf_kairos(graph_files):
    """Calculate IDF (Inverse Document Frequency) scores for nodes across graph files.

    IDF measures node rarity: high IDF = node appears in few time windows (more suspicious).

    Args:
        graph_files: List of paths to graph files

    Returns:
        tuple: (node_IDF dict mapping node labels to IDF scores, total file count)
    """
    node_set = defaultdict(set)
    for f_path in log_tqdm(graph_files, desc="Calculating IDF"):
        g = torch.load(f_path)

        f_path = f_path.split("/")[-1]
        for u, v, k in g.edges:
            node_set[g.nodes[u]["label"]].add(f_path)
            node_set[g.nodes[v]["label"]].add(f_path)

    node_IDF = {}
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(len(graph_files) / (include_count + 1))
        node_IDF[n] = IDF

    return node_IDF, len(graph_files)


def is_include_key_word_bak(s):
    """Check if node label contains benign system keywords (legacy version).

    This is an older version with slightly different keyword list than is_include_key_word().
    Used for backward compatibility with older experiments.

    Args:
        s: Node label string

    Returns:
        bool: True if label contains any benign system keyword
    """
    keywords = [
        ":",
        "null",
        "/dev/pts",
        "salt-minion.log",
        "675",
        "usr",
        "proc",
        "/.cache/mozilla/",
        "tmp",
        "thunderbird",
        "/bin/",
        "/sbin/sysctl",
        "/data/replay_logdb/",
        "/home/admin/eraseme",
        "/stat",
    ]
    flag = False
    for i in keywords:
        if i in s:
            flag = True
    return flag


def is_include_key_word(s):
    """Check if node label contains benign system keywords (current version).

    Filters out common benign nodes (system files, null values, temp files) to reduce
    false positives in queue-based investigation.

    Args:
        s: Node label string

    Returns:
        bool: True if label contains any benign system keyword
    """
    keywords = [
        ":",
        "/dev/pts",
        "salt-minion.log",
        "null",
        "usr",
        "proc",
        "firefox",
        "tmp",
        "thunderbird",
        "bin/",
        "/data/replay_logdb",
        "/stat",
        "/boot",
        "qt-opensource-linux-x64",
        "/eraseme",
        "675",
    ]
    flag = False
    for i in keywords:
        if i in s:
            flag = True
    return flag


def cal_set_rel_bak(node_IDF, s1, s2, num_dates, excluded_nodes=()):
    """Calculate queue relationship score between two node sets (legacy version).

    Counts high-IDF nodes shared between two sets to determine if queues should be merged.

    Args:
        node_IDF: Dict mapping node labels to IDF scores
        s1: First node set
        s2: Second node set
        num_dates: Total number of dates for IDF threshold calculation

    Returns:
        int: Count of shared high-IDF nodes (>90th percentile)
    """
    new_s = (
        s1 & s2
    )  # used to find common nodes in two windows to check if they should be in the same queue
    count = 0
    for i in new_s:
        #     jdata=json.loads(i)
        if i not in excluded_nodes and not is_include_key_word_bak(i):
            if i in node_IDF.keys():
                IDF = node_IDF[i]
            else:
                IDF = math.log(num_dates / (1))

            if (IDF) > math.log(
                num_dates * 0.9 / (1)
            ):  # TODO: default value for kairos, but can be parametrized
                count += 1
    return count


def cal_set_rel(
    train_node_IDF, test_node_IDF, s1, s2, num_test_dates, num_train_dates, excluded_nodes=()
):
    """Calculate queue relationship score using both train and test IDF scores.

    Improved version that considers IDF from both training and test sets for better
    anomaly detection. Queues are related if they share nodes with high combined IDF.

    Args:
        train_node_IDF: Training set IDF scores
        test_node_IDF: Test set IDF scores
        s1: First node set
        s2: Second node set
        num_test_dates: Number of test dates
        num_train_dates: Number of training dates
        excluded_nodes: Node labels to ignore (e.g. netflows)

    Returns:
        int: Count of shared high-IDF nodes (combined IDF > 5)
    """
    new_s = (
        s1 & s2
    )  # used to find common nodes in two windows to check if they should be in the same queue
    count = 0
    for i in new_s:
        if i not in excluded_nodes and not is_include_key_word(i):
            if i in test_node_IDF:
                IDF_test = test_node_IDF[i]
            else:
                IDF_test = math.log(num_test_dates / (1))

            if i in train_node_IDF:
                IDF_train = train_node_IDF[i]
            else:
                IDF_train = math.log(num_train_dates / (1))

            if (
                IDF_test + IDF_train
            ) > 5:  # TODO: default value for kairos, but can be parametrized
                count += 1
    return count


def cal_anomaly_loss_kairos(loss_list, edge_list):
    """Identify anomalous edges using statistical threshold on loss values.

    Uses mean + 1.5*std threshold (as in Kairos) to classify edges as anomalous.
    Returns nodes involved in anomalous edges and their average loss.

    Args:
        loss_list: List of loss values for edges
        edge_list: List of edge tuples (src_label, dst_label)

    Returns:
        tuple: (anomalous edge count, average anomalous loss, node_set, edge_set)
    """
    assert len(loss_list) == len(edge_list)
    count = 0
    loss_sum = 0
    edge_set = set()
    node_set = set()

    if len(loss_list) == 0:
        return count, 0.0, node_set, edge_set

    thr = mean(loss_list) + 1.5 * std(loss_list)

    for loss, (src_node, dst_node) in zip(loss_list, edge_list):
        if loss > thr:
            count += 1
            loss_sum += loss
            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add((src_node, dst_node))
    return count, loss_sum / (count + 1e-11), node_set, edge_set


def get_node_labels(cfg):
    """Returns the node ID => label mapping and the set of netflow labels.

    Labels are the same strings as the `label` attribute of the networkx graphs used to
    compute the IDF, so that both can be matched.
    """
    indexid2msg = get_indexid2msg(cfg)
    nid_to_label = {int(nid): label for nid, (_, label) in indexid2msg.items()}
    netflow_labels = {label for _, (typ, label) in indexid2msg.items() if typ == "netflow"}
    return nid_to_label, netflow_labels


def anomalous_queue_construction_kairos(
    test_tw_path, train_node_IDF, test_node_IDF, num_train_dates, num_test_dates, cfg
):
    nid_to_label, netflow_labels = get_node_labels(cfg)
    include_test_set_in_IDF = (
        cfg.evaluation.queue_evaluation.kairos_idf_queue.include_test_set_in_IDF
    )

    queues = []
    tw_names, tw_to_df = load_losses_per_time_window(test_tw_path, cfg)
    if any("srcnode" not in df.columns for df in tw_to_df.values()):
        raise ValueError(
            "Queue evaluation requires edge-level losses (srcnode/dstnode columns). "
            "It cannot be used with node-level objectives."
        )

    for tw, df in sorted(tw_to_df.items()):
        edge_loss_list = df["loss"].tolist()
        edge_list = [
            (nid_to_label.get(src, str(src)), nid_to_label.get(dst, str(dst)))
            for src, dst in zip(df["srcnode"].astype(int), df["dstnode"].astype(int))
        ]

        count, loss_avg, node_set, edge_set = cal_anomaly_loss_kairos(edge_loss_list, edge_list)
        current_tw = {
            "name": tw_names[tw],
            "loss": loss_avg,
            "index": tw,
            "nodeset": node_set,
        }

        added_que_flag = False
        for hq in queues:
            for his_tw in hq:
                if include_test_set_in_IDF:
                    cal_re = cal_set_rel(
                        train_node_IDF,
                        test_node_IDF,
                        current_tw["nodeset"],
                        his_tw["nodeset"],
                        num_test_dates,
                        num_train_dates,
                        excluded_nodes=netflow_labels,
                    )
                else:
                    cal_re = cal_set_rel_bak(
                        train_node_IDF,
                        current_tw["nodeset"],
                        his_tw["nodeset"],
                        num_train_dates,
                        excluded_nodes=netflow_labels,
                    )
                if cal_re != 0 and current_tw["name"] != his_tw["name"]:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
            if added_que_flag:
                break
        if not added_que_flag:
            queues.append([copy.deepcopy(current_tw)])

    return queues, tw_names, sorted(tw_to_df.keys())


def create_queues_kairos(cfg):
    # In kairos, IDF is computed on benign graphs (train set), optionally also on the test set
    base_dir = cfg.transformation._graphs_dir
    train_feat_files = get_all_graphs_for_dates(base_dir, cfg.dataset.train_dates)
    test_feat_files = get_all_graphs_for_dates(base_dir, cfg.dataset.test_dates)

    train_node_IDF, num_train_dates = cal_idf_kairos(train_feat_files)
    test_node_IDF, num_test_dates = cal_idf_kairos(test_feat_files)

    out_dir = cfg.evaluation.queue_evaluation._queues_dir
    os.makedirs(out_dir, exist_ok=True)

    test_losses_dir = os.path.join(cfg.training._edge_losses_dir, "test")
    for model_epoch_dir in log_tqdm(listdir_sorted(test_losses_dir), desc="Building the queues"):
        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        queues, tw_names, evaluated_tws = anomalous_queue_construction_kairos(
            test_tw_path, train_node_IDF, test_node_IDF, num_train_dates, num_test_dates, cfg
        )
        torch.save(
            {"queues": queues, "tw_names": tw_names, "evaluated_tws": evaluated_tws},
            os.path.join(out_dir, f"{model_epoch_dir}_queues.pkl"),
        )


def predict_queues(cfg):
    test_losses_dir = os.path.join(cfg.training._edge_losses_dir, "test")
    queues_dir = cfg.evaluation.queue_evaluation._queues_dir
    out_dir = cfg.evaluation.queue_evaluation._predicted_queues_dir
    os.makedirs(out_dir, exist_ok=True)
    queue_threshold = cfg.evaluation.queue_evaluation.queue_threshold

    tw_to_malicious_nodes = None
    best_stats = None
    epoch_reports = {}  # logged for the best epoch only
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        print(flush=True)
        log(console.style(epoch_name(model_epoch_dir), "bold"))
        saved = torch.load(os.path.join(queues_dir, f"{model_epoch_dir}_queues.pkl"))
        queues, tw_names, evaluated_tws = (
            saved["queues"],
            saved["tw_names"],
            saved["evaluated_tws"],
        )

        # TW windows are the same for all epochs
        if tw_to_malicious_nodes is None:
            tw_to_malicious_nodes = compute_tw_labels(cfg, tw_names=tw_names)
        labels = {tw: int(tw in tw_to_malicious_nodes) for tw in evaluated_tws}
        pred_labels = {tw: 0 for tw in evaluated_tws}
        tw_scores = {tw: 0.0 for tw in evaluated_tws}

        detected_queues = []
        report = ["Queues"]
        for queue in queues:
            is_attack = any(labels[hq["index"]] for hq in queue)
            anomaly_score = 1.0
            for hq in queue:
                anomaly_score *= hq["loss"] + 1
            idx_list = [hq["index"] for hq in queue]
            report.append(
                f"  queue {idx_list}  anomaly score {anomaly_score:.2f}"
                + (console.style("  attack", "red") if is_attack else "")
            )

            for i in idx_list:
                tw_scores[i] = max(tw_scores[i], anomaly_score)

            if anomaly_score > queue_threshold:
                detected_queues.append(idx_list)
                for i in idx_list:
                    pred_labels[i] = 1

        report += ["Detected queues", f"  {detected_queues}", "Malicious time windows"]
        torch.save(
            detected_queues, os.path.join(out_dir, f"{model_epoch_dir}_predicted_queues.pkl")
        )

        y_truth = [labels[tw] for tw in evaluated_tws]
        y_preds = [pred_labels[tw] for tw in evaluated_tws]
        scores = [tw_scores[tw] for tw in evaluated_tws]
        for tw, y_true, y_hat in zip(evaluated_tws, y_truth, y_preds):
            if y_true:
                report.append(
                    f"  {detection_mark(y_hat)}  TW {tw:<5} {tw_names[tw]}  "
                    f"queue score {tw_scores[tw]:.2f}"
                )

        stats = classifier_evaluation(y_truth, y_preds, scores)
        stats["epoch"] = int(model_epoch_dir.split("_")[-1])
        stats["num_detected_queues"] = len(detected_queues)
        epoch_reports[stats["epoch"]] = report
        log_epoch_summary(stats)
        torch.save(stats, os.path.join(out_dir, f"stats_{model_epoch_dir}.pth"))
        wandb.log(stats)

        # Kairos selects the model with best TW-level precision (then recall)
        if best_stats is None or (stats["precision"], stats["recall"]) > (
            best_stats["precision"],
            best_stats["recall"],
        ):
            best_stats = stats

    if best_stats is not None:
        log_best_epoch(
            best_stats,
            cfg,
            epoch_reports.get(best_stats["epoch"], []),
            selection="best TW-level precision, then recall",
        )
    return best_stats


def main(cfg):
    method = cfg.evaluation.queue_evaluation.used_method.strip()

    if method == "kairos_idf_queue":
        create_queues_kairos(cfg)
        return predict_queues(cfg)
    elif method == "provnet_lof_queue":
        raise NotImplementedError(
            "`provnet_lof_queue` relies on node embeddings from the removed alacarte "
            "`nodelabel2vec` files and is not supported anymore. Use `kairos_idf_queue`."
        )
    else:
        raise ValueError(f"Invalid queue evaluation method `{method}`")
