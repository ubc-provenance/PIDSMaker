import argparse

import torch
from sklearn.metrics import confusion_matrix
import logging

from provnet_utils import *
from config import *

def cal_val_thr(graph_dir):
    filelist = os.listdir(graph_dir)

    loss_list = []
    for i in sorted(filelist):
        f = open(graph_dir + i)
        for line in f:
            l = line.strip()
            jdata = eval(l)

            loss_list.append(jdata['loss'])

    thr = max(loss_list)
    # thr = np.percentile(loss_list, 90)
    logger.info(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

    return thr

def classifier_evaluation(y_test, y_test_pred):
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

    return precision,recall,fscore,accuracy,auc_val

def get_ground_truth_nids():
    ground_truth_nids = []
    with open("../Ground_Truth/E5-THEIA/THEIA-1-e5-05-15-Firefox_Drakon_APT_BinFmt_Elevate_Inject_nodes.txt", 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
    return ground_truth_nids

def ground_truth_labelling():
    ground_truth_nids = get_ground_truth_nids()

    labels = {}
    nid_to_name = {}
    filelist = os.listdir(f"{artifact_dir}//graph_5_14")
    for file in tqdm(sorted(filelist), desc="Initialize the node labels in date 05-14"):
        with open(f"{artifact_dir}//graph_5_14/{file}", 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                src_label = data['srcmsg']
                dstnode = data['dstnode']
                dst_label = data['dstmsg']
                event_time = data['time']

                if srcnode not in labels:
                    labels[srcnode] = 0

                if dstnode not in labels:
                    labels[dstnode] = 0

                nid_to_name[srcnode] = src_label
                nid_to_name[dstnode] = dst_label

    filelist = os.listdir(f"{artifact_dir}//graph_5_15")
    for file in tqdm(sorted(filelist), desc="Initialize the node labels in date 05-15"):
        with open(f"{artifact_dir}//graph_5_15/{file}", 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                src_label = data['srcmsg']
                dstnode = data['dstnode']
                dst_label = data['dstmsg']
                event_time = data['time']

                if srcnode not in labels:
                    labels[srcnode] = 0

                if dstnode not in labels:
                    labels[dstnode] = 0

                nid_to_name[srcnode] = src_label
                nid_to_name[dstnode] = dst_label

    init_labels = copy.deepcopy(labels)

    for nid in labels:
        if nid in ground_truth_nids:
            labels[nid] = 1

    return labels, init_labels, nid_to_name

def node_level_evaluation_whole_graph():
    anomalous_queue_05_14 = torch.load(f"{artifact_dir}/detected_queues_5_14")

    anomalous_queue_05_15 = torch.load(f"{artifact_dir}/detected_queues_5_15")

    labels, init_labels, nid_to_name = ground_truth_labelling()

    thr = cal_val_thr(f"{artifact_dir}/graph_5_11/")

    for queue in anomalous_queue_05_14:
        for tw in queue:
            with open(f"{artifact_dir}/graph_5_14/{tw}", 'r') as f:
                for line in f:
                    l = line.strip()
                    data = eval(l)
                    srcnode = data['srcnode']
                    dstnode = data['dstnode']
                    loss = data['loss']

                    if loss > thr:
                        init_labels[srcnode] = 1
                        init_labels[dstnode] = 1

    for queue in anomalous_queue_05_15:
        for tw in queue:
            with open(f"{artifact_dir}//graph_5_15/{tw}", 'r') as f:
                for line in f:
                    l = line.strip()
                    data = eval(l)
                    srcnode = data['srcnode']
                    dstnode = data['dstnode']
                    loss = data['loss']

                    if loss > thr:
                        init_labels[srcnode] = 1
                        init_labels[dstnode] = 1

    # Calculate the metrics
    y = []
    y_pred = []
    for nid in labels:
        y.append(labels[nid])
        y_pred.append(init_labels[nid])

    classifier_evaluation(y, y_pred)

    return labels, init_labels, nid_to_name

def record_results(
        labels,
        init_labels,
        nid_to_name
):

    # Record the FPs, FNs and TPs
    logger.info("\n\nFalse Positives:")
    for nid in labels:
        if labels[nid] == 0 and init_labels[nid] == 1:
            logger.info(f"Node: {nid_to_name[nid]}")

    logger.info("\n\nFalse Negatives:")
    for nid in labels:
        if labels[nid] == 1 and init_labels[nid] == 0:
            logger.info(f"Node: {nid_to_name[nid]}")

    logger.info("\n\nTrue Positives:")
    for nid in labels:
        if labels[nid] == 1 and init_labels[nid] == 1:
            logger.info(f"Node: {nid_to_name[nid]}")


if __name__ == "__main__":
    global logger,component_type

    logger = logging.getLogger("node_evaluation_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(artifact_dir + 'node_evaluation_with_new_labels.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Start logging.")

    labels, init_labels, nid_to_name = node_level_evaluation_whole_graph()
    record_results(labels, init_labels, nid_to_name)