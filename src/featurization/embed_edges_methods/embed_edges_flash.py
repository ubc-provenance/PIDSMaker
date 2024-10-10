from provnet_utils import *
from config import *

from torch_geometric.data import *
from tqdm import tqdm
import torch
import os

from gensim.models import Word2Vec
import math
import numpy as np

def gen_node2higvec(graph, trained_w2v_dir, indexid2props, cfg):
    w2vmodel = Word2Vec.load(os.path.join(trained_w2v_dir, "word2vec_model_final.model"))
    w2v_vector_size = cfg.featurization.embed_nodes.flash.vector_size

    all_edges = []
    for u, v, key, attr in graph.edges(data=True, keys=True):
        edge = (u, v, attr['label'], int(attr['time']))
        all_edges.append(edge)
    sorted_edges = sorted(all_edges, key=lambda x: x[3])

    nodes = {}
    for e in sorted_edges:
        src, dst, operation, t = e
        properties = [indexid2props[src] if indexid2props[src] is not None else []] + [operation] + [
            indexid2props[dst] if indexid2props[dst] is not None else []]

        if src not in nodes:
            nodes[src] = []
        if len(nodes[src]) < 300:
            nodes[src].extend(properties)

        if dst not in nodes:
            nodes[dst] = []
        if len(nodes[dst]) < 300:
            nodes[dst].extend(properties)

    node2higvec = {}
    for node, props in nodes.items():
        node2higvec[node] = infer(props, w2vmodel, PositionalEncoder(w2v_vector_size))

    return node2higvec


def infer(document, w2vmodel, encoder):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]

    embedding_dim = w2vmodel.vector_size

    if not word_embeddings:
        return np.zeros(embedding_dim)

    word_embeddings_array = np.array(word_embeddings)

    output_embedding = torch.tensor(word_embeddings_array, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

def gen_relation_onehot(rel2id):
    relvec = torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys()) // 2), num_classes=len(rel2id.keys()) // 2)
    rel2vec = {}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i] = relvec[rel2id[i] - 1]
            rel2vec[relvec[rel2id[i] - 1]] = i
    return rel2vec

def get_nid2props(cfg):
    cur, connect = init_database_connection(cfg)

    indexid2type = {}
    indexid2props = {}

    # netflow
    sql = """
            select dst_addr, index_id from netflow_node_table;
            """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        remote_ip = str(i[0])
        index_id = i[1]  # int

        indexid2type[str(index_id)] = ntype2id['netflow'] - 1 # 2
        indexid2props[str(index_id)] = remote_ip

    #subject
    sql = """
    select cmd, index_id from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        cmd = str(i[0])
        index_id = i[1]

        indexid2type[str(index_id)] = ntype2id['subject'] - 1 # 0
        indexid2props[str(index_id)] = cmd

    # file
    sql = """
       select path, index_id from file_node_table;
       """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        path = str(i[0])
        index_id = i[1]

        indexid2type[str(index_id)] = ntype2id['file'] - 1  # 1
        indexid2props[str(index_id)] = path

    return indexid2type, indexid2props

def gen_vectorized_graphs(rel2vec, node2vec, split_files, out_dir, is_test, trained_w2v_dir, indexid2props, cfg):
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    for path in tqdm(sorted_paths, desc="Computing edge embeddings"):
        file = path.split("/")[-1]

        graph = torch.load(path)

        node2higvec = gen_node2higvec(graph, trained_w2v_dir, indexid2props, cfg)

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
                node2vec[graph.nodes[u]['node_type']],
                torch.from_numpy(node2higvec[u]),
                rel2vec[attr["label"]],
                node2vec[graph.nodes[v]['node_type']],
                torch.from_numpy(node2higvec[v])
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

        os.makedirs(out_dir, exist_ok=True)
        torch.save(dataset, os.path.join(out_dir, f"{file}.TemporalData.simple"))



def main(cfg):
    log_start(__file__)

    trained_w2v_dir = cfg.featurization.embed_nodes.flash._model_dir
    graphs_dir = cfg.preprocessing.transformation._graphs_dir
    out_dir = cfg.featurization.embed_edges._edge_embeds_dir


    rel2id = get_rel2id(cfg)
    rel2vec = gen_relation_onehot(rel2id=rel2id)
    node2vec = gen_relation_onehot(rel2id=ntype2id)

    _, indexid2props = get_nid2props(cfg)

    # Vectorize training set
    gen_vectorized_graphs(
                          node2vec=node2vec,
                          rel2vec=rel2vec,
                          split_files=cfg.dataset.train_files,
                          out_dir=os.path.join(out_dir, "train/"),
                          is_test=False,
                          trained_w2v_dir=trained_w2v_dir,
                          indexid2props=indexid2props,
                          cfg=cfg,
                          )

    # Vectorize validation set
    gen_vectorized_graphs(
                          node2vec=node2vec,
                          rel2vec=rel2vec,
                          split_files=cfg.dataset.val_files,
                          out_dir=os.path.join(out_dir, "val/"),
                          is_test=False,
                          trained_w2v_dir=trained_w2v_dir,
                          indexid2props=indexid2props,
                          cfg=cfg,
                          )

    # Vectorize testing set
    gen_vectorized_graphs(
                          node2vec=node2vec,
                          rel2vec=rel2vec,
                          split_files=cfg.dataset.test_files,
                          out_dir=os.path.join(out_dir, "test/"),
                          is_test=True,
                          trained_w2v_dir=trained_w2v_dir,
                          indexid2props=indexid2props,
                          cfg=cfg,
                          )


if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)