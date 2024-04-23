import torch
from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_auc_score,
    roc_curve,
)

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

    thr = max(loss_list)
    # thr = np.percentile(loss_list, 90)
    logger.info(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

    return thr

def calculate_supervised_best_threshold(losses, labels):
    fpr, tpr, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpr, tpr)

    # Filter out the points where TPR is less than 0.84
    valid_indices = np.where(tpr >= 0.84)[0]
    fpr_valid = fpr[valid_indices]
    thresholds_valid = thresholds[valid_indices]

    # Find the threshold corresponding to the lowest FPR among valid points
    optimal_idx = np.argmin(fpr_valid)
    optimal_threshold = thresholds_valid[optimal_idx]

    return optimal_threshold

def classifier_evaluation(y_test, y_test_pred, logger):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    logger.info(f'total node num: {len(y_test)}')
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")

    logger.info("|precision|recall|fscore|accuracy|TN|FP|FN|TP|")
    logger.info(f"|{precision:.4f}|{recall:.4f}|{fscore:.4f}|{accuracy:.3f}|{tn}|{fp}|{fn}|{tp}|")

    return precision, recall, fscore, accuracy, auc_val

def get_ground_truth_nids(cfg):
    ground_truth_nids = []
    with open(os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path), 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
    return ground_truth_nids

def get_edge_index_losses_labels(tw_path, logger, cfg):
    ground_truth_nids = get_ground_truth_nids(cfg)

    logger.info(f"Loading data from {tw_path}...")
    node_labels = {}
    edge_index, losses, edge_labels = [], [], []

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

                if cfg.detection.node_evaluation.use_dst_node_loss:
                    edge_labels.append(srcnode in ground_truth_nids or dstnode in ground_truth_nids)
                else:
                    edge_labels.append(srcnode in ground_truth_nids)

                if srcnode not in node_labels:
                    node_labels[srcnode] = 0
                if dstnode not in node_labels and cfg.detection.node_evaluation.use_dst_node_loss:
                    node_labels[dstnode] = 0

    for nid in node_labels:
        if nid in ground_truth_nids:
            node_labels[nid] = 1

    return edge_index, losses, node_labels, edge_labels

def node_evaluation_without_triage(val_tw_path, tw_path, logger, cfg):
    edge_index, losses, node_labels, edge_labels = get_edge_index_losses_labels(tw_path, logger, cfg)
    
    threshold_method = cfg.detection.node_evaluation.threshold_method
    if threshold_method == "max_val_loss":
        thr = calculate_max_val_loss_threshold(val_tw_path, logger)
    elif threshold_method == "supervised_best_threshold":
        thr = calculate_supervised_best_threshold(losses, edge_labels)
    else:
        raise ValueError(f"Invalid threshold method `{threshold_method}`")
    
    pred_labels = {}
    for (srcnode, dstnode), loss, edge_label in tqdm(zip(edge_index, losses, edge_labels), desc="Edge thresholding"):
        if loss > thr:
            pred_labels[srcnode] = 1
        else:
            if srcnode not in pred_labels:
                pred_labels[srcnode] = 0

        if cfg.detection.node_evaluation.use_dst_node_loss:
            if loss > thr:
                pred_labels[dstnode] = 1
            else:
                if dstnode not in pred_labels:
                    pred_labels[dstnode] = 0

    y_truth = []
    y_pred = []
    for nid in node_labels:
        y_truth.append(node_labels[nid])
        y_pred.append(pred_labels[nid])

    logger.info(f"Start evaluating...")
    classifier_evaluation(y_truth, y_pred, logger)

def main(cfg):
    logger = get_logger(
        name="node_evaluation",
        filename=os.path.join(cfg.detection.node_evaluation._logs_dir, "node_evaluation.log"))

    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        print(f"Evaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        node_evaluation_without_triage(val_tw_path, test_tw_path, logger, cfg)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
