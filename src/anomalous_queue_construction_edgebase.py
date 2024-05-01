import argparse
import logging

import torch
from sklearn.neighbors import LocalOutlierFactor

from provnet_utils import *
from config import *


def cal_val_thr(graph_dir):
    filelist = os.listdir(graph_dir)

    loss_list = []
    for i in sorted(filelist):
        f = open(os.path.join(graph_dir, i))
        for line in f:
            l = line.strip()
            jdata = eval(l)

            loss_list.append(jdata['loss'])

    thr = max(loss_list)
    print(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

    return thr


def cal_anomaly_loss_with_val_threshold(loss_list, edge_list, thr):
    count = 0
    loss_sum = 0

    edge_set = set()
    node_set = set()

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1

            # node_info = (node_id, node_label)
            src_node_info = edge_list[i][0]
            dst_node_info = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node_info)
            node_set.add(dst_node_info)
            edge_set.add(edge_list[i][0] + edge_list[i][1])

    if count == 0:
        return 0, 0, node_set, edge_set
    else:
        return count, loss_sum / count, node_set, edge_set

# Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.

def cal_set_rel_lof(s1, s2, lof_model, nodelabels_train_val, node2vec):
    new_s = s1 & s2
    count = 0
    for i in new_s:
        # i => (node_id, node_label)
        i = i[1]
        if 'netflow' in i:
            continue

        # if path name is not in the training and validation sets
        # OutlierScore = (¬ exist) * LOF
        if i in nodelabels_train_val:
            lof_score = 0
        else:
            try:
                i = eval(i)
            except:
                print(i)
                exit()
            if "subject" in i:
                path = i['subject']
            elif "file" in i:
                path = i['file']

            try:
                emb = node2vec[path]
            except Exception as error:
                print(f"Error at {i}: {error}")
                continue
                # exit() #TODO: handle this

            lof_score = lof_model.decision_function([emb])

        if lof_score < 0:
            print(f"node:{i}, LOF score:{lof_score}")
            count += 1
    return count

def anomalous_queue_construction(
        graph_dir_path,
        lof_model = None,
        nodelabels_train_val = None,
        node2vec = None,
        val_thr = None
):
    queues = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        print(f"Time window at index {index_count}: {f_path}")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []

        # Figure out which nodes are anomalous in this time window
        for line in f:
            l = line.strip()
            jdata = eval(l)
            edge_loss_list.append(jdata['loss'])
            edge_list.append([(str(jdata['srcnode']), str(jdata['srcmsg'])), (str(jdata['dstnode']), str(jdata['dstmsg']))])

        count, loss_avg, node_set, edge_set = cal_anomaly_loss_with_val_threshold(edge_loss_list, edge_list, val_thr)
        current_tw['name'] = f_path
        current_tw['loss'] = loss_avg
        current_tw['index'] = index_count
        current_tw['nodeset'] = node_set

        # Incrementally construct the queues
        added_que_flag = False
        for hq in queues:
            for his_tw in hq:
                cal_set_rel_results = cal_set_rel_lof(current_tw['nodeset'], his_tw['nodeset'], lof_model, nodelabels_train_val, node2vec)
                if cal_set_rel_results != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            queues.append(temp_hq)

        index_count += 1

        print(f"Avg loss: {loss_avg} | Anomalies in time window: ({len(node_set)} and {len(edge_set)} malicious nodes and edges)\n")

    return queues

def train_lof_model(cfg):
    node2vec_train_val_path = os.path.join(cfg.featurization.embed_nodes._vec_graphs_dir, "nodelabel2vec_val")
    labels_and_embeddings = torch.load(node2vec_train_val_path)

    embeddings = []
    for label in labels_and_embeddings:
        # We don't consider network
        if ":" in label:
            continue
        else:
            embeddings.append(labels_and_embeddings[label])

    clf = LocalOutlierFactor(novelty=True, n_neighbors=30).fit(embeddings)
    torch.save(clf, os.path.join(cfg.detection.tw_evaluation._task_path, "trained_lof.pkl"))
    return clf

def main(cfg):
    logger = get_logger(
        name="tw_evaluation",
        filename=os.path.join(cfg.detection.tw_evaluation._logs_dir, "tw_evaluation.log"))

    node2vec_path = os.path.join(cfg.featurization.embed_nodes._vec_graphs_dir, "nodelabel2vec")
    node2vec_train_val_path = os.path.join(cfg.featurization.embed_nodes._vec_graphs_dir, "nodelabel2vec_val")
    
    node2vec = torch.load(node2vec_path)
    nodelabels_train_val = list(torch.load(node2vec_train_val_path).keys())

    lof_model = train_lof_model(cfg)

    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        # Threshold
        val_thr = cal_val_thr(val_tw_path)

        # Testing date
        queues = anomalous_queue_construction(
            graph_dir_path=test_tw_path,
            lof_model=lof_model,
            nodelabels_train_val=nodelabels_train_val,
            node2vec=None,
            val_thr=val_thr
        )
        
        out_dir = cfg.detection.tw_evaluation._queues_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(queues, os.path.join(out_dir, f"{model_epoch_dir}_queues.pkl"))


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)
    
    main(cfg)
