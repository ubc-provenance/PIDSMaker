import argparse
import logging

import torch

from provnet_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'anomalous_queue.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
    logger.info(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

    return thr


def cal_anomaly_loss_with_val_threshold(loss_list, edge_list, thr):
    count = 0
    loss_sum = 0

    edge_set = set()
    node_set = set()

    logger.info(f"thr:{thr}")

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
                print(i)
                print(path)
                print(error)
                exit()

            lof_score = lof_model.decision_function([emb])

        if lof_score < 0:
            logger.info(f"node:{i}, LOF score:{lof_score}")
            count += 1
    return count

def anomalous_queue_construction(
        graph_dir_path,
        lof_model = None,
        nodelabels_train_val = None,
        node2vec = None,
        val_thr = None
):
    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        if "2019-05-14" in f_path or "2019-05-15" in f_path:
            nodefilelist = os.listdir(w2v_models_dir)
            for n in nodefilelist:
                if 'nodelabel2vec_' + f_path[:19] in n:
                    node2vec = torch.load(w2v_models_dir + f"{n}")
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        logger.info(f'Time window index: {index_count}')

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
        for hq in history_list:
            for his_tw in hq:
                cal_set_rel_results = cal_set_rel_lof(current_tw['nodeset'], his_tw['nodeset'], lof_model,
                                                      nodelabels_train_val, node2vec)
                if cal_set_rel_results != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1


        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-node2vec_path', help='Input the path to stored node2vec maps from Word2Vec', required=False)
    parser.add_argument('-node2vec_train_val_path', help='Input the path to stored node2vec (train and val datasets only) maps from Word2Vec.', required=False)
    parser.add_argument('-lof_path', help='Input the path to trained lof model', required=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logger.info("Start logging.")

    args = get_args()
    node2vec_path = args.node2vec_path
    node2vec_train_val_path = args.node2vec_train_val_path
    lof_path = args.lof_path

    node2vec = torch.load(node2vec_path)
    nodelabels_train_val = list(torch.load(node2vec_train_val_path).keys())
    lof_model = torch.load(lof_path)

    # Validation date
    val_thr = cal_val_thr(f"{artifact_dir}/graph_5_11/")

    # history_list = anomalous_queue_construction(
    #     graph_dir_path=f"{artifact_dir}/graph_5_11/",
    #     lof_model=lof_model,
    #     nodelabels_train_val=nodelabels_train_val,
    #     node2vec=node2vec,
    #     val_thr=val_thr
    # )
    # torch.save(history_list, f"{artifact_dir}/graph_5_11_history_list")


    # Testing date
    history_list = anomalous_queue_construction(
        graph_dir_path=f"{artifact_dir}/graph_5_14/",
        lof_model=lof_model,
        nodelabels_train_val=nodelabels_train_val,
        node2vec=None,
        val_thr=val_thr
    )
    torch.save(history_list, f"{artifact_dir}/graph_5_14_history_list")

    history_list = anomalous_queue_construction(
        graph_dir_path=f"{artifact_dir}/graph_5_15/",
        lof_model=lof_model,
        nodelabels_train_val=nodelabels_train_val,
        node2vec=None,
        val_thr=val_thr
    )
    torch.save(history_list, f"{artifact_dir}/graph_5_15_history_list")
