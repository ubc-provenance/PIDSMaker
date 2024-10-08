from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from nltk.tokenize import word_tokenize

from provnet_utils import *
from config import *
import torch
import numpy as np
import random

def splitting_label_set(split_files: list[str], cfg):
    base_dir = cfg.preprocessing.transformation._graphs_dir
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
            tagged_data: list[str],
            model_save_path: str,
            indexid2msg: dict,
            epochs: int,
            emb_dim: int,
            alpha: float,
            min_alpha: float,
            dm: int = 1,
):
    SEED = 0
    model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED)
    model.build_vocab(tagged_data)

    for epoch in range(epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=1, compute_loss=True)
        model.alpha -= 0.0002
        if model.alpha < min_alpha:
            model.alpha = min_alpha
        log(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')

    log(f'Saving Doc2Vec model to {model_save_path}')
    model.save(os.path.join(model_save_path, 'doc2vec_model.model'))


# Used in Rcaid
def tokenize_using_neighbors_features(graph_list):
    documents = []
    nodes = set()
    for idx, G in enumerate(graph_list):
        # Prepare the training data for Doc2Vec: each node and its neighbors as a 'document'
        for node in G.nodes():
            if node not in nodes:
                node_label = G.nodes[node].get('label', '')  # Assumes each node has a 'label' attribute
                neighbors = list(G.neighbors(node))
                neighbor_labels = []
                nodes.add(node)
                for neighbor in neighbors:
                    label = G.nodes[neighbor].get('label', '')
                    if isinstance(label, list):
                        label = ' '.join(label)
                    neighbor_labels.append(label)

                document = node_label + ' ' + ' '.join(neighbor_labels)
                documents.append(TaggedDocument(words=document.split(), tags=[str(node)]))

    return documents

def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.embed_nodes.doc2vec._model_dir
    os.makedirs(model_save_path,exist_ok=True)

    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    train_files = cfg.dataset.train_files
    
    # Context-aware Doc2vec embedding that considers the neighbors when creating embedding (like in Rcaid)
    if cfg.featurization.embed_nodes.doc2vec.include_neighbors:
        sorted_paths = get_all_files_from_folders(cfg.preprocessing.transformation._graphs_dir, train_files)
        graph_list = []
        for path in tqdm(sorted_paths, desc='Loading graphs'):
            graph = torch.load(path)
            graph_list.append(graph)
        tagged_data = tokenize_using_neighbors_features(graph_list)
    
    # Standard token-level Doc2vec
    else:
        train_set_nodes = splitting_label_set(split_files=train_files, cfg=cfg)
        words, tags = preprocess(indexid2msg, train_set_nodes)
        tagged_data = [TaggedDocument(words=word_list, tags=[tag]) for word_list, tag in zip(words, tags)]

    epochs = cfg.featurization.embed_nodes.doc2vec.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.doc2vec.alpha
    min_alpha = cfg.featurization.embed_nodes.doc2vec.min_alpha

    log(f"Start building and training Doc2Vec model...")
    doc2vec(tagged_data=tagged_data,
                  model_save_path=model_save_path,
                  indexid2msg=indexid2msg,
                  epochs=epochs,
                  emb_dim=emb_dim,
                  alpha=alpha,
                  min_alpha=min_alpha,
                  cfg=cfg)

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
