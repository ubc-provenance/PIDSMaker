import os
import torch

from config import *
from provnet_utils import get_device, set_seed, log_start, log
from data_utils import load_all_datasets


def get_preprocessed_graphs(cfg):
    log("Loading preprocessed graphs...")
    out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
    out_file = os.path.join(out_dir, "torch_graphs.pkl")
    return torch.load(out_file)


def main(cfg):
    set_seed(cfg)
    log_start(__file__)
    
    device = get_device(cfg)
    train_data, val_data, test_data, max_node_num = load_all_datasets(cfg, device)
    
    out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
    out_file = os.path.join(out_dir, "torch_graphs.pkl")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Saving preprocessed graphs to {out_file}...")
    torch.save((train_data, val_data, test_data, max_node_num), out_file)
