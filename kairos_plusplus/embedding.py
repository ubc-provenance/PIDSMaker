from torch_geometric.data import *
from tqdm import tqdm

import torch
import os

from config import *
from provnet_utils import *

def gen_relation_onehot(rel2id):
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    return rel2vec

def gen_vectorized_graphs(node2higvec, rel2vec, split_files, out_dir, is_test, trained_w2v_dir, logger, cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    include_edge_type = cfg.featurization.embed_edges.include_edge_type

    for path in tqdm(sorted_paths, desc="Computing edge embeddings"):
        file = path.split("/")[-1]
        if is_test:
            node2higvec = torch.load(os.path.join(trained_w2v_dir, f"nodelabel2vec_test-{file}"))

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
            if include_edge_type:
                msg.append(torch.cat([
                    torch.from_numpy(node2higvec[graph.nodes[u]['label']]),
                    rel2vec[attr["label"]],
                    torch.from_numpy(node2higvec[graph.nodes[v]['label']])
                ]))
            else:
                msg.append(torch.cat([
                    torch.from_numpy(node2higvec[graph.nodes[u]['label']]),
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

        os.makedirs(out_dir, exist_ok=True)
        torch.save(dataset, os.path.join(out_dir, f"{file}.TemporalData.simple"))

        logger.info(f'Graph: {file}. Events num: {len(sorted_edges)}. Node num: {len(graph.nodes)}')


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    logger = get_logger(
        name="embed_edges",
        filename=os.path.join(cfg.featurization.embed_nodes._logs_dir, "embed_edges.log"))

    trained_w2v_dir = cfg.featurization.embed_nodes._vec_graphs_dir
    graphs_dir = cfg.preprocessing.build_graphs._graphs_dir
    out_dir = cfg.featurization.embed_edges._edge_embeds_dir

    node2higvec = torch.load(os.path.join(trained_w2v_dir, "nodelabel2vec_val")) # From both train and val
    rel2vec = gen_relation_onehot(rel2id=rel2id)
    
    # Vectorize training set
    gen_vectorized_graphs(node2higvec=node2higvec,
                          rel2vec=rel2vec,
                          split_files=cfg.dataset.train_files,
                          out_dir=os.path.join(out_dir, "train/"),
                          is_test=False,
                          trained_w2v_dir=trained_w2v_dir,
                          logger=logger,
                          cfg=cfg,
                        )

    # Vectorize validation set
    gen_vectorized_graphs(node2higvec=node2higvec,
                          rel2vec=rel2vec,
                          split_files=cfg.dataset.val_files,
                          out_dir=os.path.join(out_dir, "val/"),
                          is_test=False,
                          trained_w2v_dir=trained_w2v_dir,
                          logger=logger,
                          cfg=cfg,
                        )

    # Vectorize testing set
    gen_vectorized_graphs(node2higvec=node2higvec,
                            rel2vec=rel2vec,
                            split_files=cfg.dataset.test_files,
                            out_dir=os.path.join(out_dir, "test/"),
                            is_test=True,
                            trained_w2v_dir=trained_w2v_dir,
                            logger=logger,
                            cfg=cfg,
                            )
