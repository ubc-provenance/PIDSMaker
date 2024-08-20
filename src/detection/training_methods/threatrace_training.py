from provnet_utils import *
from config import *

import torch.nn.functional as F
from threatrace_utils.model import SAGENet
from threatrace_utils.data_process import load_train_graph, TestDataset
from torch_geometric.data import NeighborSampler, DataLoader
import torch

from tqdm import tqdm
import os

def train(model, loader, optimizer, data, device):
    model.train()
    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()

def train_pro(cfg):
    model_save_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(model_save_dir, exist_ok=True)

    log(f"Get training args")
    model_name = cfg.detection.gnn_training.threatrace.model
    b_size = cfg.detection.gnn_training.threatrace.batch_size
    thre = cfg.detection.gnn_training.threatrace.thre
    lr = cfg.detection.gnn_training.threatrace.lr
    weight_decay = cfg.detection.gnn_training.threatrace.weight_decay
    epochs = cfg.detection.gnn_training.threatrace.epochs

    split_files = cfg.dataset.train_files
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    device = get_device(cfg)

    log("Initial model")
    model = SAGENet(10 * 2, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(epochs), desc="Training model"):
        for graph in sorted_paths:
            data1, feature_num, label_num, adj, adj2, node_list = load_train_graph(graph)
            dataset = TestDataset(data1)
            data = dataset[0]
            loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False,
                                     add_self_loops=True)
            loss = train(model, loader, optimizer, data, device)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_{epoch}.pth'))
        log(f"Model of epoch {epoch} is saved")

def main(cfg):
    train_pro(cfg)
    log(f"Finish gnn_training")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)