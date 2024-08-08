from provnet_utils import *
from config import *

import os
from gensim.models import Word2Vec
from flash_utils.models import PositionalEncoder, infer, GCN
from flash_utils.utils import load_one_graph_data, get_nid2props
import torch

from sklearn.utils import class_weight
from torch.nn import CrossEntropyLoss
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

def main(cfg):
    model_save_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(model_save_dir, exist_ok=True)

    word2vec_model_dir = cfg.featurization.embed_nodes.flash._model_dir
    w2vmodel = Word2Vec.load(os.path.join(word2vec_model_dir, "word2vec_model_final.model"))

    w2v_vector_size = cfg.featurization.embed_nodes.flash.vector_size

    device = get_device(cfg)

    in_channel = cfg.detection.gnn_training.flash.in_channel
    out_channel = cfg.detection.gnn_training.flash.out_channel
    lr = cfg.detection.gnn_training.flash.lr
    weight_decay = cfg.detection.gnn_training.flash.weight_decay
    epochs = cfg.detection.gnn_training.flash.epochs

    model = GCN(in_channel, out_channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    split_files = cfg.dataset.train_files
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    indexid2type, indexid2props = get_nid2props(cfg)

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(sorted_paths)):
            phrases, labels, edges, mapp = load_one_graph_data(sorted_paths[i], indexid2type, indexid2props)
            l = np.array(labels)
            class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(l), y=l)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')

            nodes = [infer(x, w2vmodel, PositionalEncoder(w2v_vector_size)) for x in phrases]
            nodes = np.array(nodes)

            graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device),
                         y=torch.tensor(labels, dtype=torch.long).to(device),
                         edge_index=torch.tensor(edges, dtype=torch.long).to(device))

            graph.n_id = torch.arange(graph.num_nodes)
            mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000, input_nodes=mask)
            for subg in loader:
                model.train()
                optimizer.zero_grad()
                out = model(subg.x, subg.edge_index)
                loss = criterion(out, subg.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * subg.batch_size
            log(total_loss / mask.sum().item())

            loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000, input_nodes=mask)
            for subg in loader:
                model.eval()
                out = model(subg.x, subg.edge_index)

                sorted, indices = out.sort(dim=1, descending=True)
                conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
                # conf = (conf - conf.min()) / conf.max()
                conf = (conf - conf.min()) / conf.max() if conf.max() > 0 else conf  # Handle division by zero

                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.9)

                # Ensure subg.n_id[cond] is on the same device as mask
                subg_n_id = subg.n_id.to(device)
                mask[subg_n_id[cond]] = False

            log(f'Model# {epoch} and graph {i}/{len(sorted_paths)}. {mask.sum().item()} nodes still misclassified \n')

        torch.save(model.state_dict(), os.path.join(model_save_dir,f'lword2vec_gnn_{epoch}.pth'))


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)