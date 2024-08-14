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

def main(cfg):
    checkpoints_dir = cfg.featurization.embed_edges.magic._magic_checkpoints_dir

    device = get_device(cfg)
    log(f"Magic testing on device: {device}")

    log("Get testing args")
    train_args = CN()
    train_args.num_hidden = cfg.featurization.embed_edges.magic.num_hidden
    train_args.num_layers = cfg.featurization.embed_edges.magic.num_layers
    train_args.max_epoch = cfg.featurization.embed_edges.magic.max_epoch
    train_args.negative_slope = cfg.featurization.embed_edges.magic.negative_slope
    train_args.mask_rate = cfg.featurization.embed_edges.magic.mask_rate
    train_args.alpha_l = cfg.featurization.embed_edges.magic.alpha_l
    train_args.optimizer = cfg.featurization.embed_edges.magic.optimizer
    train_args.lr = cfg.featurization.embed_edges.magic.lr
    train_args.weight_decay = cfg.featurization.embed_edges.magic.weight_decay

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
        tw_to_nids_xtest = {}
        for tw in range(n_test):
            tw_to_nids_xtest[tw] = {}

            g, tw_name, nid_list = load_entity_level_dataset(t='test', n=tw, cfg=cfg)
            g.to(device)

            tw_to_nids_xtest[tw]['nids'] = nid_list
            tw_to_nids_xtest[tw]['x_test'] = model.embed(g).cpu().numpy()

            del g

    log("Get results")
    tw_score = evaluate_entity_level_using_knn(x_train=x_train,tw_to_nids_xtest=tw_to_nids_xtest)

    #save tw_score

    out_dir = cfg.detection.gnn_testing._magic_test_dir
    os.makedirs(out_dir, exist_ok=True)

    torch.save(tw_score, os.path.join(out_dir, "tw_score.pth"))

def evaluate_entity_level_using_knn(x_train, tw_to_nids_xtest):

    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std

    x_train = cudf.DataFrame.from_records(x_train)
    log(f"x_train DataFrame created with shape: {x_train.shape}")

    n_neighbors = 10

    log("Initialize and train KNN")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(x_train)

    log("Get mean distance of training data")
    idx = list(range(x_train.shape[0]))
    random.shuffle(idx)

    # Add debug information
    log(f"Randomly shuffled indices: {idx[:10]}")  # Log the first 10 indices for reference
    log(f"x_train shape: {x_train.shape}, idx length: {len(idx)}")

    try:
        distances_train, _ = nbrs.kneighbors(x_train.iloc[idx[:min(50000, x_train.shape[0])]].to_pandas(),
                                             n_neighbors=n_neighbors)
    except KeyError as e:
        log(f"KeyError encountered: {e}")
        log(f"Available columns in x_train: {x_train.columns}")
        raise
    del x_train
    mean_distance_train = distances_train.mean().mean()
    del distances_train

    if mean_distance_train == 0:
        log("Warning: mean_distance_train is zero, setting it to a small value to avoid division by zero")
        mean_distance_train = 1e-9

    log("Get scores of testing nodes")
    scores_list = []
    tw_score = {}
    for tw, data in tqdm(tw_to_nids_xtest.items()):
        x_test = data['x_test']
        x_test = cudf.DataFrame.from_records(x_test)
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        distances = distances.mean(axis=1)

        # Ensure distances and mean_distance_train are numpy arrays for division
        distances = distances.to_numpy()
        # mean_distance_train = mean_distance_train.to_numpy()

        score = distances / mean_distance_train
        tw_score[tw] = score
        scores_list.append(score)

        del distances

    return tw_score




