from torch_geometric.data import *
from tqdm import tqdm

import logging
import torch
import os

from config import *
from provnet_utils import *

# Setting for logging
logger = logging.getLogger(f"embedding_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + f'embedding.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def gen_relation_onehot(rel2id):
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    torch.save(rel2vec, artifact_dir + f"rel2vec")
    return rel2vec

def gen_vectorized_graphs(node2higvec, rel2vec, g_dir, saved_vec_g_dir):
    for file in tqdm(sorted(os.listdir(g_dir)), desc="Vectorizing the graphs"):
        path = g_dir + file
        graph = torch.load(path)

        sorted_edges = sorted(graph.edges(data=True, keys=True), key=lambda t: t[3]["time"])

        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        for u, v, k, attr in sorted_edges:
            src.append(int(u))
            dst.append(int(v))
            msg.append(torch.cat([
                torch.from_numpy(node2higvec[graph.nodes[u]['label']]),
                rel2vec[attr["label"]],
                torch.from_numpy(node2higvec[graph.nodes[v]['label']])
            ]))
            t.append(int(attr["time"]))

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)

        os.system(f"mkdir -p {saved_vec_g_dir}/")
        torch.save(dataset, f"{saved_vec_g_dir}/" + file + ".TemporalData.simple")

        logger.info(f'Graph: {file}. Events num: {len(sorted_edges)}. Node num: {len(graph.nodes)}')

def gen_vectorized_test_graphs(rel2vec, g_dir, saved_vec_g_dir):
    for file in tqdm(sorted(os.listdir(g_dir)), desc="Vectorizing the graphs"):
        node2higvec = torch.load(w2v_models_dir + f"nodelabel2vec_{file}")
        path = g_dir + file
        graph = torch.load(path)

        sorted_edges = sorted(graph.edges(data=True, keys=True), key=lambda t: t[3]["time"])

        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        for u, v, k, attr in sorted_edges:
            src.append(int(u))
            dst.append(int(v))
            msg.append(torch.cat([
                torch.from_numpy(node2higvec[graph.nodes[u]['label']]),
                rel2vec[attr["label"]],
                torch.from_numpy(node2higvec[graph.nodes[v]['label']])
            ]))
            t.append(int(attr["time"]))

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = torch.tensor(t)
        dataset.msg = torch.vstack(msg)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)

        os.system(f"mkdir -p {saved_vec_g_dir}/")
        torch.save(dataset, f"{saved_vec_g_dir}/" + file + ".TemporalData.simple")

        logger.info(f'Graph: {file}. Events num: {len(sorted_edges)}. Node num: {len(graph.nodes)}')

if __name__ == "__main__":
    os.system(f"mkdir -p {vec_graphs_dir}")

    logger.info("Start logging.")

    cur, _ = init_database_connection()

    node2higvec = torch.load(w2v_models_dir + f"nodelabel2vec_train_val")
    rel2vec = gen_relation_onehot(rel2id=rel2id)

    saved_vec_g_dir = vec_graphs_dir
    os.system(f"mkdir -p {saved_vec_g_dir}/")

    # Vectorize training set
    gen_vectorized_graphs(node2higvec=node2higvec,
                          rel2vec=rel2vec,
                          g_dir=f"{graphs_dir}/train/",
                          saved_vec_g_dir=f"{saved_vec_g_dir}/train/")

    # Vectorize validation set
    gen_vectorized_graphs(node2higvec=node2higvec,
                          rel2vec=rel2vec,
                          g_dir=f"{graphs_dir}/val/",
                          saved_vec_g_dir=f"{saved_vec_g_dir}/val/")

    # Vectorize testing set
    gen_vectorized_test_graphs(rel2vec=rel2vec,
                               g_dir=f"{graphs_dir}/test/",
                               saved_vec_g_dir=f"{saved_vec_g_dir}/test/")

