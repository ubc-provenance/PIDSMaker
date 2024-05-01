import argparse

import torch
from sklearn.metrics import confusion_matrix
import logging

from provnet_utils import *
from config import *
from model import *


def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    print(f'tn: {tn}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"fscore: {fscore}")
    print(f"accuracy: {accuracy}")
    print(f"auc_val: {auc_val}")

    print("|precision|recall|fscore|accuracy|TN|FP|FN|TP|")
    print(f"|{precision:.4f}|{recall:.4f}|{fscore:.4f}|{accuracy:.3f}|{tn}|{fp}|{fn}|{tp}|")

    return precision,recall,fscore,accuracy,auc_val

def ground_truth_label(test_tw_path):
    labels = {}
    for tw in os.listdir(test_tw_path):
        labels[tw] = 0

    attack_list = [ # TODO: parametrize
            # tgn + 2MLP streaming
            '2018-04-10 14:18:50.641713026~2018-04-10 14:34:12.001897298.txt',
            '2018-04-10 14:34:12.001969452~2018-04-10 14:49:12.508890812.txt',
            '2018-04-10 14:49:12.510151372~2018-04-10 15:04:20.772713218.txt',
            '2018-04-10 15:04:20.772732545~2018-04-10 15:20:57.001950599.txt',
            '2018-04-11 13:41:59.957267640~2018-04-11 13:57:04.138342417.txt'

        ]

    for i in attack_list:
        if i in labels:
            labels[i] = 1

    return labels

def main(cfg):
    print(f"threshold 1: {beta_day10}")
    print(f"threshold 2: {beta_day11}")

    # Evaluating the testing set
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        
        pred_label = {}
        for tw in os.listdir(test_tw_path):
            pred_label[tw] = 0
        
        queues = torch.load(os.path.join(cfg.detection.tw_evaluation._queues_dir, f"{model_epoch_dir}_queues.pkl"))
        labels = ground_truth_label(test_tw_path)

        detected_queues = []
        for queue in queues:
            label = any([labels[hq["name"]] for hq in queue ])
            anomaly_score = 0
            for hq in queue:
                if anomaly_score == 0:
                    anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
                else:
                    anomaly_score = (anomaly_score) * (hq['loss'] + 1)
            print(f"-> queue anomaly score: {anomaly_score:.2f} | {'ATTACK' if label else ''}")
            
            if anomaly_score > beta_day10:
                name_list = []
                for i in queue:
                    name_list.append(i['name'])
                print(f"Anomalous queue: {name_list}")
                detected_queues.append(name_list)
                for i in name_list:
                    pred_label[i] = 1
                print(f"Anomaly score: {anomaly_score}")
        
        out_dir = cfg.detection.tw_evaluation._predicted_queues_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(detected_queues, os.path.join(out_dir, f"{model_epoch_dir}_predicted_queues.pkl"))

        # Calculate the metrics
        print("\n********************************* Attack Labels *********************************")
        y = []
        y_pred = []
        for i in labels:
            y.append(labels[i])
            y_pred.append(pred_label[i])
        classifier_evaluation(y, y_pred)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)
    
    main(cfg)
