import os

import torch
from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.embed_nodes_methods.embed_paths_provd import get_node2corpus
from pidsmaker.utils import get_indexid2msg, log_start, log_tqdm


def main(cfg):
    log_start(__file__)

    indexid2msg = get_indexid2msg(cfg)
    node2corpus = get_node2corpus(
        splits=["test"], cfg=cfg
    )  # no GNN encoding so we only need test set

    doc2vec_model_path = os.path.join(
        cfg.featurization.embed_nodes._model_dir, "doc2vec_model.model"
    )
    model = Doc2Vec.load(doc2vec_model_path)

    path_emb2nodes = {}
    path2vector_cache = {}

    # Create an embedding for each path in the dataset and associate the nodes that are from these paths
    for indexid, _ in log_tqdm(indexid2msg.items(), desc="Embeding all paths in the dataset"):
        corpus = node2corpus[indexid]
        for path in corpus:
            path_str = str(path)
            if path_str not in path2vector_cache:
                model.random.seed(0)
                vector = model.infer_vector(path)
                path2vector_cache[path_str] = vector

            vector = path2vector_cache[path_str]
            if path_str not in path_emb2nodes:
                path_emb2nodes[path_str] = {"vector": vector, "nodes": set()}

            path_emb2nodes[path_str]["nodes"].add(indexid)

    model_save_dir = cfg.featurization.embed_edges._model_dir
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(dict(path_emb2nodes), os.path.join(model_save_dir, "path_emb2nodes.pkl"))
