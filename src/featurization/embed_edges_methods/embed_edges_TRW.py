import os
from provnet_utils import *
from config import *

from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from torch_geometric.data import *
import torch

def cal_word_weight(n,percentage):
    d = -1 / n * percentage / 100
    a_1 = 1/n - 0.5 * (n-1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence

def get_indexid2vec(indexid2msg, model_path, use_node_types, decline_percentage, node_list):
    model = Word2Vec.load(model_path)

    indexid2vec = {}
    # for indexid, msg in tqdm(indexid2msg.items(), desc='processing indexid2vec:'):
    for indexid in tqdm(node_list, desc='processing indexid2vec:'):
        msg = indexid2msg[int(indexid)]
        if msg[0] == 'subject':
            if use_node_types:
                tokens = tokenize_subject(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_subject(msg[1])
        elif msg[0] == 'file':
            if use_node_types:
                tokens = tokenize_file(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_file(msg[1])
        else:
            if use_node_types:
                tokens = tokenize_netflow(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_netflow(msg[1])

        weight_list = cal_word_weight(len(tokens), decline_percentage)
        word_vectors = [model.wv[word] for word in tokens]
        weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
        sentence_vector = np.mean(weighted_vectors, axis=0)

        normalized_vector = sentence_vector / np.linalg.norm(sentence_vector)

        indexid2vec[int(indexid)] = np.array(normalized_vector)


    return indexid2vec

def gen_relation_onehot(rel2id):
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    return rel2vec

def gen_vectorized_graphs(indexid2vec, etype2oh, ntype2oh, split_files, out_dir, cfg):
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    for path in tqdm(sorted_paths, desc="Computing edge embeddings"):
        file = path.split("/")[-1]

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
                ntype2oh[graph.nodes[u]['node_type']],
                torch.from_numpy(indexid2vec[int(u)]),
                etype2oh[attr["label"]],
                ntype2oh[graph.nodes[v]['node_type']],
                torch.from_numpy(indexid2vec[int(v)])
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
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, (cfg.dataset.train_files +
                                                         cfg.dataset.test_files +
                                                         cfg.dataset.val_files))
    used_nodes = set()
    for file_path in tqdm(sorted_paths, desc="Get nodes in graphs:"):
        graph = torch.load(file_path)
        used_nodes = used_nodes | set(graph.nodes())
    used_nodes = list(used_nodes)

    use_node_types = cfg.featurization.embed_nodes.temporal_rw.use_node_types
    use_cmd = cfg.featurization.embed_nodes.temporal_rw.use_cmd
    use_port = cfg.featurization.embed_nodes.temporal_rw.use_port
    decline_rate = cfg.featurization.embed_nodes.temporal_rw.decline_rate

    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)

    trw_word2vec_model_path = cfg.featurization.embed_nodes.temporal_rw._model_dir + 'trw_word2vec.model'
    indexid2vec = get_indexid2vec(indexid2msg=indexid2msg, model_path=trw_word2vec_model_path,
                                  use_node_types=use_node_types, decline_percentage=decline_rate,
                                  node_list=used_nodes)

    rel2id = get_rel2id(cfg)

    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)

    # Vectorize training set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.train_files,
                          out_dir=os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "train/"),
                          cfg=cfg
                          )

    # Vectorize validation set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.val_files,
                          out_dir=os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "val/"),
                          cfg=cfg
                          )

    # Vectorize testing set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.test_files,
                          out_dir=os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "test/"),
                          cfg=cfg
                          )

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)