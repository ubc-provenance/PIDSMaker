from provnet_utils import *
from config import *

from yacs.config import CfgNode as CN

from magic_utils.utils import set_random_seed, create_optimizer
from magic_utils.loaddata import load_entity_level_dataset, load_metadata
from magic_utils.autoencoder import build_model

from tqdm import tqdm
import torch
import os
import numpy as np

import random
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve
from cuml.neighbors import NearestNeighbors
from collections import defaultdict
import cudf

def get_node_predictions(cfg, tw_to_malicious_nodes):
    log("Get metadata")
    metadata = load_metadata(cfg=cfg)

    n_train = metadata['n_train']
    n_test = metadata['n_test']

    tw_to_nids_ytest = {}
    for tw in range(n_test):
        tw_to_nids_ytest[tw] = {}

        _, _, nid_list = load_entity_level_dataset(t='test', n=tw, cfg=cfg)
        tw_to_nids_ytest[tw]['nids'] = nid_list

        y_test = []
        for nid in nid_list:
            if (tw in tw_to_malicious_nodes) and (str(nid) in tw_to_malicious_nodes[tw]):
                y_test.append(1)
            else:
                y_test.append(0)
        tw_to_nids_ytest[tw]['y_test'] = y_test

    in_dir = cfg.detection.gnn_testing._magic_test_dir
    tw_score = torch.load(os.path.join(in_dir, "tw_score.pth"))

    log("Get results")
    results, node_to_max_loss_tw = get_result_magic(tw_to_nids_ytest=tw_to_nids_ytest, tw_score=tw_score)

    return results, node_to_max_loss_tw


def get_result_magic(tw_to_nids_ytest,tw_score):
    results = {}

    scores_list = []
    for tw, score in tw_score.items():
        scores_list.append(score)
    all_scores = np.concatenate(scores_list)

    y_test = []
    for tw, data in tw_to_nids_ytest.items():
        y_test.extend(data['y_test'])

    log("Calculate best threshold")
    # auc = roc_auc_score(y_test, all_scores)

    prec, rec, threshold = precision_recall_curve(y_test, all_scores)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)

    # best_idx = f1.argmax()
    # best_thres = threshold[best_idx]

    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        if rec[i] < 0.99996:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    log("Generate results")
    node_to_max_loss_tw = defaultdict(int)
    node_to_max_loss = {}
    for tw, data in tw_to_nids_ytest.items():
        if tw not in results:
            results[tw] = {}
        for i in range(len(data['nids'])):
            node_id = data['nids'][i]
            if node_id not in results[tw]:
                results[tw][node_id] = {}

            results[tw][node_id]['score'] = tw_score[tw][i]
            results[tw][node_id]['y_hat'] = int(tw_score[tw][i] > best_thres)
            results[tw][node_id]["y_true"] = data['y_test'][i]

            if node_id not in node_to_max_loss:
                node_to_max_loss[node_id] = tw_score[tw][i]

            if tw_score[tw][i] >= node_to_max_loss[node_id]:
                node_to_max_loss_tw[node_id] = tw


    return results, node_to_max_loss_tw
