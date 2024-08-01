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
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

def get_node_predictions(cfg, tw_to_malicious_nodes):
    checkpoints_dir = cfg.featurization.embed_nodes.magic._magic_checkpoints_dir

    device = get_device(cfg)

    log("Get training args")
    train_args = CN()
    train_args.num_hidden = cfg.featurization.embed_nodes.magic.num_hidden
    train_args.num_layers = cfg.featurization.embed_nodes.magic.num_layers
    train_args.max_epoch = cfg.featurization.embed_nodes.magic.max_epoch
    train_args.negative_slope = cfg.featurization.embed_nodes.magic.negative_slope
    train_args.mask_rate = cfg.featurization.embed_nodes.magic.mask_rate
    train_args.alpha_l = cfg.featurization.embed_nodes.magic.alpha_l
    train_args.optimizer = cfg.featurization.embed_nodes.magic.optimizer
    train_args.lr = cfg.featurization.embed_nodes.magic.lr
    train_args.weight_decay = cfg.featurization.embed_nodes.magic.weight_decay

    set_random_seed(0)

    log("Get metadata")
    metadata = load_metadata(cfg=cfg)
    train_args.n_dim = metadata['node_feature_dim']
    train_args.e_dim = metadata['edge_feature_dim']

    log("Build model")
    model = build_model(train_args, device)
    model.load_state_dict(torch.load(checkpoints_dir + "checkpoints.pt", map_location=device))
    model = model.to(device)
    model.eval()

    n_train = metadata['n_train']
    n_test = metadata['n_test']

    with torch.no_grad():

        log("Get x_train")
        x_train = []
        for i in range(n_train):
            g, tw_name, nid_list = load_entity_level_dataset(t='train', n=i, cfg=cfg)
            g.to(device)
            x_train.append(model.embed(g).cpu().numpy())
            del g
        x_train = np.concatenate(x_train, axis=0)

        log("Get testing data")
        tw_to_nids_xtest_ytest = {}
        for tw in range(n_test):
            tw_to_nids_xtest_ytest[tw] = {}

            g, tw_name, nid_list = load_entity_level_dataset(t='test', n=i, cfg=cfg)
            g.to(device)

            tw_to_nids_xtest_ytest[tw]['nids'] = nid_list
            tw_to_nids_xtest_ytest[tw]['x_test'] = model.embed(g).cpu().numpy()
            y_test = []
            for nid in nid_list:
                if (tw in tw_to_malicious_nodes) and (str(nid) in tw_to_malicious_nodes[tw]):
                    y_test.append(1)
                else:
                    y_test.append(0)
            tw_to_nids_xtest_ytest[tw]['y_test'] = y_test
            del g

    log("Get results")
    results, node_to_max_loss_tw = evaluate_entity_level_using_knn(x_train=x_train, tw_to_nids_xtest_ytest=tw_to_nids_xtest_ytest)

    return results, node_to_max_loss_tw


def evaluate_entity_level_using_knn(x_train, tw_to_nids_xtest_ytest):
    results = {}

    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std

    n_neighbors = 10

    log("Initialize and train KNN")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    log("Get mean distance of training data")
    idx = list(range(x_train.shape[0]))
    random.shuffle(idx)
    distances_train, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
    del x_train
    mean_distance_train = distances_train.mean()
    del distances_train

    log("Get scores of testing nodes")
    y_test = []
    scores_list = []
    tw_score = {}
    for tw, data in tw_to_nids_xtest_ytest.items():
        x_test = data['x_test']
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        distances = distances.mean(axis=1)
        score = distances / mean_distance_train
        tw_score[tw] = score
        scores_list.append(score)
        y_test.extend(data['y_test'])
        del distances
    all_scores = np.concatenate(scores_list)

    log("Calculate best threshold")
    # auc = roc_auc_score(y_test, all_scores)

    prec, rec, threshold = precision_recall_curve(y_test, all_scores)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)

    best_idx = f1.argmax()  # 获取 F1 分数的最大值的索引
    best_thres = threshold[best_idx]

    log("Generate results")
    node_to_max_loss_tw = defaultdict(int)
    node_to_max_loss = {}
    for tw, data in tw_to_nids_xtest_ytest.items():
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

































