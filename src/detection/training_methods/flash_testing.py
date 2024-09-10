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

from torch_geometric import utils

def main(cfg, model, epoch):
    result_dir = cfg.detection.gnn_training._edge_losses_dir
    os.makedirs(result_dir, exist_ok=True)

    word2vec_model_dir = cfg.featurization.embed_nodes.flash._model_dir
    w2vmodel = Word2Vec.load(os.path.join(word2vec_model_dir, "word2vec_model_final.model"))

    w2v_vector_size = cfg.featurization.embed_nodes.flash.vector_size

    device = get_device(cfg)

    in_channel = cfg.detection.gnn_training.flash.in_channel
    out_channel = cfg.detection.gnn_training.flash.out_channel

    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    split_files = cfg.dataset.test_files
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    indexid2type, indexid2props = get_nid2props(cfg)

    tw_to_result = {}
    log(f"Start testing epoch {epoch} in device {device}")

    for i in range(len(sorted_paths)):
        tw_to_result[i] = {}
        tw_to_result[i]['nids'] = []
        tw_to_result[i]['score'] = []
        tw_to_result[i]['y_hat'] = []

        phrases, labels, edges, mapp = load_one_graph_data(sorted_paths[i], indexid2type, indexid2props)

        nodes = [infer(x, w2vmodel, PositionalEncoder(w2v_vector_size)) for x in phrases]
        nodes = np.array(nodes)

        graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device),
                        y=torch.tensor(labels, dtype=torch.long).to(device),
                        edge_index=torch.tensor(edges, dtype=torch.long).to(device))
        graph.n_id = torch.arange(graph.num_nodes)
        flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool).to(device)

        loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000)

        for subg in loader:
            model.eval()
            out = model(subg.x, subg.edge_index)

            sorted, indices = out.sort(dim=1, descending=True)
            conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
            # conf = (conf - conf.min()) / conf.max()
            conf = (conf - conf.min()) / conf.max() if conf.max() > 0 else conf  # Handle division by zero

            pred = indices[:, 0]
            cond = (pred == subg.y) & (conf > 0.53)

            cond = cond.to(device)
            # Ensure subg.n_id[cond] is on the same device as mask
            subg_n_id = subg.n_id.to(device)

            flag[subg_n_id[cond]] = torch.logical_and(flag[subg_n_id[cond]],
                                                        torch.tensor([False] * len(flag[subg_n_id[cond]]),dtype=torch.bool).to(device))

            index = utils.mask_to_index(flag).tolist()
            MP_ids = [mapp[x] for x in index]
            MP_set = set(MP_ids)

            subg_n_ids = subg.n_id.tolist()
            subg_actual_ids = [mapp[x] for x in subg_n_ids]


            tw_to_result[i]['nids'].extend(subg_actual_ids)
            tw_to_result[i]['score'].extend(conf.tolist())
            tw_to_result[i]['y_hat'].extend(1 if x in MP_set else 0 for x in subg_actual_ids)


        log(f'Model# {epoch} and graph {i}/{len(sorted_paths)} evaluation finished.')

    out_file = os.path.join(result_dir, f'tw_to_mp{epoch}.pth')
    torch.save(tw_to_result, out_file)
    log(f"Model positive nodes are saved in {out_file}")

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
