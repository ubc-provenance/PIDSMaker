import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from gensim.models import Doc2Vec
from provnet_utils import *
from flash_utils.utils import load_one_graph_data, get_nid2props
from encoders import GATModel


# Training loop
def train(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data)
    loss = criterion(out, data.y)  # Assuming data.y contains labels
    loss.backward()
    optimizer.step()
    return loss.item()

def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

def load_one_graph_data(graph_path, indexid2type, indexid2props,cfg):
    nx_g = torch.load(graph_path)

    all_edges = []
    for u, v, key, attr in nx_g.edges(data=True, keys=True):
        edge = (u, v, attr['label'], int(attr['time']))
        all_edges.append(edge)
    sorted_edges = sorted(all_edges, key=lambda x: x[3])
    model = Doc2Vec.load(cfg.featurization.embed_nodes.doc2vec._model_dir)
    nodes, labels, edges = {}, {}, []
    for e in sorted_edges:
        src, dst, operation, t = e
        properties = [indexid2props[src] if indexid2props[src] is not None else []] + [operation] + [
            indexid2props[dst] if indexid2props[dst] is not None else []]

        if src not in nodes:
            nodes[src] = model.infer_vector([src])

        labels[src] = indexid2type[src]

        if dst not in nodes:
            nodes[dst] = model.infer_vector([dst])

        labels[dst] = indexid2type[dst]

        edges.append((src, dst))

    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map)

    return features, feat_labels, edge_index, list(index_map.keys())

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    in_channels = cfg.detection.gnn_training.in_channels  # Number of input features per node
    hidden_dim = cfg.detection.gnn_training.node_hid_dim  # Number of hidden units in GAT layers and MLP
    out_channels = cfg.detection.gnn_training.node_out_dim  # Number of output classes for classification
    lr = cfg.detection.gnn_training.lr
    num_epochs = 200

    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    split_files = cfg.dataset.train_files
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    indexid2type, indexid2props = get_nid2props(cfg)


    # Initialize model, optimizer, and loss function
    model = GATModel(in_channels=in_channels,out_channels=out_channels,hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Create sample graph data (replace this with your actual graph dataset)
    model.train()
    for epoch in range(100):
        for i in range(len(sorted_paths)):
            features, feat_labels, edge_index, node_ids = load_one_graph_data(sorted_paths[i], indexid2type, indexid2props,cfg)
            x = torch.tensor(features, dtype=torch.float).to(device)
            y = torch.tensor(feat_labels, dtype=torch.long).to(device)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

            data = Data(x=x, edge_index=edge_index, y=y).to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            yy = out
            for i in range(len(out)):
                yy[i] = 0
            loss = criterion(out, yy)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f'new_model_epoch_{epoch}.pth')