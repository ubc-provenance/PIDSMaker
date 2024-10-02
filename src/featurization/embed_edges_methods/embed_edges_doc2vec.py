import os
from provnet_utils import *
from config import *

from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tqdm import tqdm
from torch_geometric.data import *

def get_indexid2vec(indexid2msg, model_path):

    model = Doc2Vec.load(model_path)

    indexid2vec = {}
    for indexid, msg in tqdm(indexid2msg.items(), desc='processing indexid2vec:'):
        if msg[0] == 'subject':
            tokens = tokenize_subject(msg[1])
        if msg[0] == 'file':
            tokens = tokenize_file(msg[1])
        if msg[0] == 'netflow':
            tokens = tokenize_netflow(msg[1])

        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)

        indexid2vec[int(indexid)] = normalized_vector


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
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
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
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    doc2vec_model_path = cfg.featurization.embed_nodes.doc2vec._model_dir + 'doc2vec_model.model'
    indexid2vec = get_indexid2vec(indexid2msg, doc2vec_model_path)

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