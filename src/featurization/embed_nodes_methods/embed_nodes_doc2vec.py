from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from nltk.tokenize import word_tokenize

from provnet_utils import *
from config import *
import torch
import numpy as np
import random

def splitting_label_set(split_files: list[str], cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    node_set = set()

    for path in sorted_paths:
        graph = torch.load(path)
        nodes = graph.nodes()
        node_set = node_set | set(nodes)

    return list(node_set)  #list[str]

def preprocess(indexid2msg: dict, nodes: list[str]):

    tags = []
    words = []

    for node in nodes:
        node_type, msg = indexid2msg[int(node)]
        tags.append(node)

        if node_type == 'subject':
            words.append(tokenize_subject(msg))
        if node_type == 'file':
            words.append(tokenize_file(msg))
        if node_type == 'netflow':
            words.append(tokenize_netflow(msg))

    return words, tags


def doc2vec(cfg,
            train_set: list[str],
            model_save_path: str,
            indexid2msg: dict,
            epochs: int,
            emb_dim: int,
            alpha: float,
            min_alpha: float,
            dm: int = 1):
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    words, tags = preprocess(indexid2msg, train_set)
    tagged_data = [TaggedDocument(words=word_list, tags=[tag]) for word_list, tag in zip(words, tags)]

    if use_seed:
        model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED)
    else:
        model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True)
    model.build_vocab(tagged_data)


    for epoch in range(epochs):
        model.train(tagged_data, total_examples=len(words), epochs=1, compute_loss=True)
        model.alpha -= 0.0002
        if model.alpha < min_alpha:
            model.alpha = min_alpha
        log(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')

    log(f'Saving Doc2Vec model to {model_save_path}')
    model.save(model_save_path + 'doc2vec_model.model')
    pass

def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.embed_nodes.doc2vec._model_dir
    os.makedirs(model_save_path,exist_ok=True)

    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    train_set_nodes = splitting_label_set(split_files=cfg.dataset.train_files, cfg=cfg)
    # val_set_nodes = splitting_label_set(split_files=cfg.dataset.val_files, cfg=cfg)
    # test_set_nodes = splitting_label_set(split_files=cfg.dataset.test_files, cfg=cfg)

    epochs = cfg.featurization.embed_nodes.doc2vec.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.doc2vec.alpha
    min_alpha = cfg.featurization.embed_nodes.doc2vec.min_alpha

    log(f"Start building and training Doc2Vec model...")
    doc2vec(train_set=train_set_nodes,
                  model_save_path=model_save_path,
                  indexid2msg=indexid2msg,
                  epochs=epochs,
                  emb_dim=emb_dim,
                  alpha=alpha,
                  min_alpha=min_alpha,
                  cfg=cfg)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
