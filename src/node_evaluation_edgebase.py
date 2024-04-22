import argparse

import torch
from sklearn.metrics import confusion_matrix
import logging

from provnet_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 're_evaluation_without_triage.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def cal_val_thr(graph_dir):
    filelist = list_files_in_directory(graph_dir)

    loss_list = []
    for file in sorted(filelist):
        f = open(file)
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

    return precision, recall, fscore, accuracy, auc_val

def get_ground_truth_nids():
    ground_truth_nids = []
    with open("../Ground_Truth/E5-THEIA/THEIA-1-e5-05-15-Firefox_Drakon_APT_BinFmt_Elevate_Inject_nodes_new.txt", 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
    return ground_truth_nids

def node_evaluation_without_triage(thr, datapath):
    ground_truth_nids = get_ground_truth_nids()

    logger.info("Loading data...")
    labels = {}
    pred_labels ={}

    filelist = list_files_in_directory(datapath)
    for file in tqdm(sorted(filelist), desc="Initialize the node labels in test set"):
        logger.info(f"Loading data from file: {file}")
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                dstnode = data['dstnode']

                loss = data['loss']

                if srcnode not in labels:
                    labels[srcnode] = 0

                if dstnode not in labels:
                    labels[dstnode] = 0

                if loss > thr:
                    pred_labels[srcnode] = 1
                else:
                    if srcnode not in pred_labels:
                        pred_labels[srcnode] = 0

                if loss > thr:
                    pred_labels[dstnode] = 1
                else:
                    if dstnode not in pred_labels:
                        pred_labels[dstnode] = 0

    logger.info(f"Labelling...")
    for nid in labels:
        if nid in ground_truth_nids:
            labels[nid] = 1

    y_truth = []
    y_pred = []
    for nid in labels:
        y_truth.append(labels[nid])
        y_pred.append(pred_labels[nid])

    logger.info(f"Start evaluating...")
    classifier_evaluation(y_truth, y_pred)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='input embedding type (re, bce or hybrid)', required=True)
    args = parser.parse_args()

    if args.embedding == 're':
        datapath = re_test_dir + 'test/'
        valpath = re_test_dir + 'val/'
    elif args.embedding == 'hybrid':
        datapath = hybrid_test_dir + 'test/'
        valpath = hybrid_test_dir + 'val/'


    logger.info("Start logging.")
    thr = cal_val_thr(valpath)
    node_evaluation_without_triage(thr, datapath)
