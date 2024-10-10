import csv
import os
from tqdm import tqdm
from encoders import TGNEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch
from joblib import load

def testing(clf, test_pv,cfg):
    split_files = cfg.dataset.test_files
    test_files = get_all_files_from_folders(graph_dir, split_files)
    test_graphs = {}
    for path in tqdm(test_files, desc="Computing edge embeddings"):
        graph = torch.load(path)
        test_g.append(graph)
    i = 0
    for file in test_files:
        test_graphs[file] = test_g[i]
        i += 1
    for file_name in test_graphs:
        # Detection
        graph = test_graphs[file_name]
        [path_vectors, top_uncommon_paths, text_list] = test_pv[file_name]
        path_pred = clf.predict(path_vectors)
        path_scores = clf.decision_function(path_vectors)
        prediction_results = [path_pred, path_scores]
        outputs = []
        for i in range(len(path_pred)):
            if path_pred[i] == -1:
                outputs.append(top_uncommon_paths[i][0])
        outputs = str(outputs)
        pred = {}
        for nid in graph.nodes:
            # ignore the pseudo node
            if graph.nodes[nid]["prov_type"] == "pseudo node":
                continue
            # the node is in the path which was flagged as anomalous by ProvDetector
            if nid in outputs:
                pred[nid] = 1
            else:
                pred[nid] = 0
        fieldnames = ['node_id', 'prediction']

        logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
        os.makedirs(logs_dir, exist_ok=True)
        output_csv_path = os.path.join(logs_dir, file_name + ".csv")
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for nid, prediction in pred.items():
                writer.writerow({'node_id': nid, 'prediction': prediction})


    return 0

def main(cfg):
    model_path = cfg.detection.gnn_training._trained_models_dir + "provd_model.pkl"
    clf = load(model_path)
    test_pv = cfg.featurization.embed_edges._edge_embeds_dir + "testembeddings.npy"
    testing(clf,test_pv,cfg)
