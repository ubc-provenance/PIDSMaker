import torch
from provnet_utils import *
from config import *
import os

from tqdm import tqdm
from threatrace_utils.model import SAGENet
from threatrace_utils.data_process import load_train_graph
from torch_geometric.loader import NeighborSampler, DataLoader, NeighborLoader
import torch.nn.functional as F

def test_pro(cfg):
    model_save_dir = cfg.detection.gnn_training._trained_models_dir

    log(f"Get training args")
    model_name = cfg.detection.gnn_training.threatrace.model
    b_size = cfg.detection.gnn_training.threatrace.batch_size
    thre = cfg.detection.gnn_training.threatrace.thre
    lr = cfg.detection.gnn_training.threatrace.lr
    weight_decay = cfg.detection.gnn_training.threatrace.weight_decay
    epochs = cfg.detection.gnn_training.threatrace.epochs

    split_files = cfg.dataset.test_files
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    device = get_device(cfg)

    model = SAGENet(10 * 2, 3).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'model_{epochs - 1}.pth')))
    model.eval()

    tw_node_data = {}
    for tw,graph in tqdm(enumerate(sorted_paths), desc="Testing model"):
        tw_node_data[tw] = {}

        data, feature_num, label_num, adj, adj2, node_list = load_train_graph(graph)
        data.to(device)

        loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=b_size, shuffle=False, input_nodes=data.test_mask)

        for data_flow in loader:
            score_list = []

            data_flow = data_flow.to(device)
            out = model(data_flow.x, data_flow.edge_index)

            pred = out.max(1)[1]
            pro = F.softmax(out, dim=1)
            pro1 = pro.max(1)
            for i in range(len(data_flow.n_id)):
                pro[i][pro1[1][i]] = -1
            pro2 = pro.max(1)
            for i in range(len(data_flow.n_id)):
                if pro2[0][i] != 0:
                    score_list.append(pro1[0][i] / pro2[0][i])
                else:
                    score_list.append(pro1[0][i] / 1e-5)
            for i in range(len(data_flow.n_id)):
                node = node_list[data_flow.n_id[i]]
                score = score_list[i]
                y_hat = int((data.y[data_flow.n_id[i]] == pred[i]) and score > thre)

                if node not in tw_node_data[tw]:
                    tw_node_data[tw][node] = {}
                    tw_node_data[tw][node]['score'] = 0
                    tw_node_data[tw][node]['y_hat'] = 0

                if score > tw_node_data[tw][node]['score']:
                    tw_node_data[tw][node]['score'] = score

                tw_node_data[tw][node]['y_hat'] = int(tw_node_data[tw][node]['y_hat'] or y_hat)

    out_dir = cfg.detection.gnn_testing._threatrace_test_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tw_node_data, os.path.join(out_dir, f"tw_node_data.pth"))

def main(cfg):
    log_start(__file__)
    test_pro(cfg)
    log("Finish gnn_testing")

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)