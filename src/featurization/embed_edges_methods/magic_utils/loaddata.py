import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os

from provnet_utils import *
from config import *
import torch

def transform_graph(g, node_feature_dim, edge_feature_dim):
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def preload_entity_level_dataset(cfg):
    magic_file_dir = cfg.preprocessing.build_graphs._magic_dir
    os.makedirs(magic_file_dir, exist_ok=True)

    if os.path.exists(magic_file_dir + '/metadata.json'):
        with open(magic_file_dir + '/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    else:
        magic_graph_dir = cfg.preprocessing.build_graphs.magic_graphs_dir
        train_nx_filelist = get_all_files_from_folders(magic_graph_dir, cfg.dataset.train_files)
        test_nx_filelist = get_all_files_from_folders(magic_graph_dir, cfg.dataset.test_files)

        n_train = len(train_nx_filelist)
        n_test = len(test_nx_filelist)

        node_feature_dim = 0
        edge_feature_dim = 0

        for g_file in train_nx_filelist + test_nx_filelist:
            nx_g = torch.load(g_file)
            g = dgl.from_networkx(
                nx_g,
                node_attrs=['type'],
                edge_attrs=['type'],
            )
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)

        node_feature_dim += 1
        edge_feature_dim += 1

        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'n_train': n_train,
            'n_test': n_test,
        }
        with open(magic_file_dir + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)

        return metadata


def load_metadata(cfg):
    return preload_entity_level_dataset(cfg=cfg)


def load_entity_level_dataset(t, n, cfg):
    metadata = preload_entity_level_dataset(cfg=cfg)
    node_feature_dim = metadata['node_feature_dim']
    edge_feature_dim = metadata['edge_feature_dim']

    if t == "train":
        split_files = cfg.dataset.train_files
    elif t == "test":
        split_files = cfg.dataset.test_files

    magic_graph_dir = cfg.preprocessing.build_graphs.magic_graphs_dir
    sorted_paths = get_all_files_from_folders(magic_graph_dir, split_files)
    file_path = sorted_paths[n]

    nx_g = torch.load(file_path)
    dgl_g = dgl.from_networkx(
        nx_g,
        node_attrs=['type'],
        edge_attrs=['type'],
    )
    g = transform_graph(dgl_g, node_feature_dim, edge_feature_dim)
    return g, file_path.split('/')[-1]