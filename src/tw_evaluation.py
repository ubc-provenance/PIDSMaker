import argparse

import torch
from sklearn.metrics import confusion_matrix
import logging

from provnet_utils import *
from config import *
from model import *


# Setting for logging
logger = logging.getLogger("tw_evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'tw_evaluation_with_new_labels.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
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

def ground_truth_label():
    labels = {}
    filelist = os.listdir(f"{artifact_dir}/graph_5_14")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_5_15")
    for f in filelist:
        labels[f] = 0

    attack_list = [
            # graphtransformer + 1MLP streaming
            # '2018-04-10 13:31:11.753518345~2018-04-10 13:46:38.001612377.txt',
            # '2018-04-10 14:18:50.641683397~2018-04-10 14:34:12.001897298.txt',
            # '2018-04-10 14:34:12.001897298~2018-04-10 14:49:12.508890812.txt',
            # '2018-04-10 14:49:12.508890812~2018-04-10 15:04:20.772713218.txt',
            # '2018-04-10 15:04:20.772713218~2018-04-10 15:20:57.001950599.txt',

            # tgn + 2MLP streaming
            '2019-05-15 14:44:51.126952463~2019-05-15 15:00:27.769327074.txt',

        ]

    for i in attack_list:
        if i in labels:
            labels[i] = 1

    return labels

def evaluate_10_11():
    logger.info(f"threshold 1: {beta_day10}")
    logger.info(f"threshold 2: {beta_day11}")

    # Validation date
    history_list = torch.load(f"{artifact_dir}/graph_5_11_history_list")
    anomalous_queue_scores = []
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        for i in hl:
            name_list.append(i['name'])
        anomalous_queue_scores.append(anomaly_score)
    logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")

    # Evaluating the testing set
    pred_label = {}

    filelist = os.listdir(f"{artifact_dir}/graph_5_14/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_5_15/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{artifact_dir}/graph_5_14_history_list")
    detected_queues = []
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        if anomaly_score > beta_day10:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            detected_queues.append(name_list)
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")
    torch.save(detected_queues, f"{artifact_dir}/detected_queues_5_14")

    history_list = torch.load(f"{artifact_dir}/graph_5_15_history_list")
    detected_queues = []
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day11:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            detected_queues.append(name_list)
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")
    torch.save(detected_queues, f"{artifact_dir}/detected_queues_5_15")

    # history_list = torch.load(f"{artifact_dir}/test_history_list")
    # detected_queues = []
    # for hl in history_list:
    #     anomaly_score = 0
    #     for hq in hl:
    #         if anomaly_score == 0:
    #             anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
    #         else:
    #             anomaly_score = (anomaly_score) * (hq['loss'] + 1)
    #     if anomaly_score > beta_day10:
    #         name_list = []
    #         for i in hl:
    #             name_list.append(i['name'])
    #         logger.info(f"Anomalous queue: {name_list}")
    #         detected_queues.append(name_list)
    #         for i in name_list:
    #             pred_label[i] = 1
    #         logger.info(f"Anomaly score: {anomaly_score}")
    # torch.save(detected_queues, f"{artifact_dir}/detected_queues_test")


    # Calculate the metrics
    logger.info("\n********************************* Attack Labels *********************************")
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])
    classifier_evaluation(y, y_pred)

if __name__ == "__main__":
    logger.info("Start logging.")

    logger.info("\n\n************************* Evaluate 10/11 *************************")
    evaluate_10_11()