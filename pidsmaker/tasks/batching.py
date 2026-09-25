import os

import torch

from pidsmaker.utils.data_utils import load_all_datasets
from pidsmaker.utils.utils import get_device, log, log_start, set_seed


def get_preprocessed_graphs(cfg):
    if not cfg._save_graph_preprocessing:
        log("Force graphs loading to save storage.")
        return get_data(cfg)

    log("Loading preprocessed graphs...")
    out_dir = cfg.batching._preprocessed_graphs_dir
    out_file = os.path.join(out_dir, "torch_graphs.pkl")
    return torch.load(out_file)

def get_data(cfg):
    device = get_device(cfg)
    return load_all_datasets(cfg, device)

def main(cfg):
    set_seed(cfg)
    log_start(__file__)
    
    if not cfg._save_graph_preprocessing:
        log("Skipping task to save storage.")
        return

    train_data, val_data, test_data, max_node_num = get_data(cfg)

    out_dir = cfg.batching._preprocessed_graphs_dir
    out_file = os.path.join(out_dir, "torch_graphs.pkl")
    os.makedirs(out_dir, exist_ok=True)
    log(f"Saving preprocessed graphs to {out_file}...")
    torch.save((train_data, val_data, test_data, max_node_num), out_file)
