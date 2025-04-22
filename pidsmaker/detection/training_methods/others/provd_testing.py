import os
from collections import defaultdict

import numpy as np
import torch

from pidsmaker.utils.utils import get_all_files_from_folders, log


def main(cfg):
    clf = torch.load(os.path.join(cfg.featurization.feat_training._model_dir, "lof.pkl"))
    path_emb2nodes = torch.load(
        os.path.join(cfg.featurization.feat_inference._model_dir, "path_emb2nodes.pkl")
    )

    nodes, vectors = [], []
    for path, d in path_emb2nodes.items():
        nodes.append(d["nodes"])
        vectors.append(d["vector"])

    log("Running LOF inference...")
    vectors = np.array(vectors)
    preds = clf.predict(vectors)
    scores = clf.decision_function(vectors)

    # here, a node is malicious if it is within a malicious path
    malicious_nodes = set()
    node_to_max_score = defaultdict(int)
    for nodes, pred, score in zip(nodes, preds, scores):
        if pred == -1:
            malicious_nodes |= nodes
        for node in nodes:
            node_to_max_score[node] = max(node_to_max_score[node], score)

    graph_dir = cfg.preprocessing.transformation._graphs_dir
    test_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)
    test_graphs = [torch.load(g) for g in test_paths]
    test_nodes = set()
    for g in test_graphs:
        test_nodes |= set(g.nodes())

    node_list = []
    for node in test_nodes:
        node_list.append(
            {
                "node": int(node),
                "score": node_to_max_score[node],
                "y_hat": int(node in malicious_nodes),
            }
        )

    out_dir = os.path.join(cfg.featurization.feat_inference._model_dir, "node_list.pkl")
    torch.save(node_list, out_dir)
