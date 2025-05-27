import os

import torch

from pidsmaker.utils.data_utils import load_all_datasets
from pidsmaker.utils.utils import get_device, log, log_start, set_seed


def get_preprocessed_graphs(cfg):
    if cfg.detection.graph_preprocessing.save_on_disk:
        log("Loading preprocessed graphs...")
        out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
        out_file = os.path.join(out_dir, "torch_graphs.pkl")
        train_data, val_data, test_data, max_node_num = torch.load(out_file)

    else:
        log("Computing graphs...")
        device = get_device(cfg)
        train_data, val_data, test_data, max_node_num = load_all_datasets(cfg, device)

    return train_data, val_data, test_data, max_node_num


def main(cfg):
    set_seed(cfg)
    log_start(__file__)

    if cfg.detection.graph_preprocessing.save_on_disk:
        device = get_device(cfg)
        train_data, val_data, test_data, max_node_num = load_all_datasets(cfg, device)

        out_dir = cfg.detection.graph_preprocessing._preprocessed_graphs_dir
        out_file = os.path.join(out_dir, "torch_graphs.pkl")
        os.makedirs(out_dir, exist_ok=True)
        log(f"Saving preprocessed graphs to {out_file}...")
        torch.save((train_data, val_data, test_data, max_node_num), out_file)

    else:
        log("Not saving to disk, skipping this task.")
