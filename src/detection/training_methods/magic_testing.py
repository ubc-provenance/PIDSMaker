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

def main(cfg, model, n_train, n_test, epoch):
    device = get_device(cfg)
    set_random_seed(0)

    model = model.to(device)
    model.eval()


    with torch.no_grad():

        log("Get x_train")
        x_train = []
        for i in range(n_train):
            g, tw_name, nid_list = load_entity_level_dataset(t='train', n=i, cfg=cfg)
            g.to(device)
            x_train.append(model.embed(g).cpu().numpy())
            del g
        x_train = np.concatenate(x_train, axis=0)

        num_nodes = x_train.shape[0]
        sample_size = 50000 * 2
        # log(f"x_train shape: {x_train.shape}")
        sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
        x_train_sampled = x_train[sample_indices]
        # log(f"x_train_sampled shape: {x_train_sampled.shape}")

        del x_train

        x_train_mean = x_train_sampled.mean(axis=0)
        x_train_std = x_train_sampled.std(axis=0)
        x_train_sampled = (x_train_sampled - x_train_mean) / x_train_std

        torch.cuda.empty_cache()

        x_train_sampled = cudf.DataFrame.from_records(x_train_sampled)
        # log(f"x_train DataFrame created with shape: {x_train_sampled.shape}")

        n_neighbors = 10

        # log("Initialize and train KNN")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train_sampled)

        # log("Get mean distance of training data")
        idx = list(range(x_train_sampled.shape[0]))
        random.shuffle(idx)

        try:
            distances_train, _ = nbrs.kneighbors(x_train_sampled.iloc[idx[:min(50000, x_train_sampled.shape[0])]].to_pandas(),
                                                 n_neighbors=n_neighbors)
        except KeyError as e:
            log(f"KeyError encountered: {e}")
            log(f"Available columns in x_train: {x_train_sampled.columns}")
            raise
        del x_train_sampled
        mean_distance_train = distances_train.mean().mean()
        del distances_train

        if mean_distance_train == 0:
            log("Warning: mean_distance_train is zero, setting it to a small value to avoid division by zero")
            mean_distance_train = 1e-9

        torch.cuda.empty_cache()

        log("Get testing results")
        # scores_list = []
        tw_score = {}
        for tw in tqdm(range(n_test),desc="testing time windows"):
            g, tw_name, nid_list = load_entity_level_dataset(t='test', n=tw, cfg=cfg)
            g.to(device)

            x_test = model.embed(g).cpu().numpy()
            del g

            x_test = cudf.DataFrame.from_records(x_test)

            distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
            del x_test

            distances = distances.mean(axis=1)

            # Ensure distances and mean_distance_train are numpy arrays for division
            distances = distances.to_numpy()
            # mean_distance_train = mean_distance_train.to_numpy()

            score = distances / mean_distance_train
            tw_score[tw] = score
            # scores_list.append(score)

            del distances
            torch.cuda.empty_cache()


    #save tw_score

    out_dir = cfg.detection.gnn_training._edge_losses_dir
    os.makedirs(out_dir, exist_ok=True)

    torch.save(tw_score, os.path.join(out_dir, f"tw_score{epoch}.pth"))
