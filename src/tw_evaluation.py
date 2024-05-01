import torch
import re
import wandb

from provnet_utils import *
from config import *


def ground_truth_label(test_tw_path):
    labels = {}
    for tw in listdir_sorted(test_tw_path):
        labels[tw] = 0

    attack_tws = cfg.dataset.ground_truth_time_windows
    for tw in attack_tws:
        # TODO: this is a workaround, should be replaced with exact event timestamps
        label_tw_match = [label for label in labels.keys() if tw.startswith(label[:19])]
        assert len(label_tw_match) > 0, f"Attack time window file not found: {tw}"
        
        matched_label_tw = label_tw_match[0]
        labels[matched_label_tw] = 1

    return labels

def main(cfg):
    print(f"threshold 1: {beta_day10}") # TODO: remove
    print(f"threshold 2: {beta_day11}")

    # Evaluating the testing set
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    
    best_precision, best_stats = 0.0, None
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        
        pred_label = {}
        for tw in listdir_sorted(test_tw_path):
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
        y, y_pred = [], []
        for i in labels:
            y.append(labels[i])
            y_pred.append(pred_label[i])

        stats = classifier_evaluation(y, y_pred, y_pred)
        stats["epoch"] = int(re.findall(r'[+-]?\d*\.?\d+', model_epoch_dir)[0])
        wandb.log(stats)
        
        if precision > best_precision:
            best_precision = precision
            best_stats = stats
        
    wandb.log(best_stats)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)
    
    main(cfg)
