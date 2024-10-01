from provnet_utils import *
from config import *
from factory import *

import torch.nn.functional as F
from threatrace_utils.data_process import load_train_graph, TestDataset
from torch_geometric.loader import NeighborSampler, DataLoader, NeighborLoader
import torch

from tqdm import tqdm
import os
from . import threatrace_testing

def train(encoder, decoder, loader, optimizer, data, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    for data_flow in loader:
        data_flow = data_flow.to(device)
        optimizer.zero_grad()
        out = encoder(data_flow.x, data_flow.edge_index)
        loss = decoder(out, data.y[data_flow.n_id].to(device), inference=False)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()

def train_pro(cfg):
    b_size = cfg.detection.gnn_training.threatrace.batch_size
    epochs = cfg.detection.gnn_training.num_epochs

    split_files = cfg.dataset.train_files
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    device = get_device(cfg)

    train_data = [load_train_graph(graph)[0] for graph in sorted_paths]
    max_node_num = max([data.x.max().item() for data in train_data]) + 1
    # model = build_model(data_sample=train_data[0], device=device, cfg=cfg, max_node_num=max_node_num)
    
    data_sample = train_data[0]
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = None
    encoder = encoder_factory(cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim, graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num)
    decoder = decoder_factory(cfg, in_dim=in_dim, device=device, max_node_num=max_node_num)
    optimizer = optimizer_factory(cfg, (encoder.parameters() | decoder[0].parameters()))

    for epoch in tqdm(range(epochs), desc="Training model"):
        for data in train_data:
            data.to(device)

            loader = batch_loader_factory(cfg, data, graph_reindexer)
            loss = train(encoder, decoder, loader, optimizer, data, device)
        # torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_{epoch}.pth'))
        # log(f"Model of epoch {epoch} is saved")
        log(f"Testing for epoch {epoch}")
        threatrace_testing.main(model, epoch, cfg)

def main(cfg):
    log_start(__file__)
    train_pro(cfg)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)