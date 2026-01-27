from itertools import chain

import torch
from gensim.models.doc2vec import TaggedDocument

from pidsmaker.utils.utils import (
    get_all_files_from_folders,
    get_indexid2msg,
    get_split2nodes,
    log_tqdm,
    tokenize_label,
)


def get_splits_to_train_featurization(cfg):
    """
    Returns the splits on which train the embedding method.
    """
    training_split = cfg.featurization.training_split.strip()
    if training_split == "all":
        return ["train", "val", "test"]

    if training_split == "train":
        return ["train"]

    raise ValueError(f"Invalid training split {training_split}")


def get_corpus(cfg, doc2vec_format=False, gather_multi_dataset=False):
    """
    Returns the tokenized labels of all nodes within a specific split.
    For training, the train split can be used to get only the corpus of train set.
    """
    splits = get_splits_to_train_featurization(cfg)
    split2nodes = get_split2nodes(cfg, gather_multi_dataset=gather_multi_dataset)
    nodes_to_include = set().union(*(split2nodes[split] for split in splits))

    words, tags = [], []
    node_labels = set()
    indexid2msg = get_indexid2msg(cfg, gather_multi_dataset=gather_multi_dataset)

    for node, msg in indexid2msg.items():
        node_type, node_label = msg

        if (node in nodes_to_include) and (node_label not in node_labels):
            node_labels.add(node_label)
            tags.append(node)

            words.append(tokenize_label(node_label, node_type))

    if doc2vec_format:
        words = [
            TaggedDocument(words=word_list, tags=[str(tag)]) for word_list, tag in zip(words, tags)
        ]

    return words


# Used in Rcaid
def get_corpus_using_neighbors_features(cfg, doc2vec_format=False):
    """
    Same but also adds the tokens of neighbors in the final tokens.
    We need to loop on the graphs here to find neighbors.
    """
    splits = get_splits_to_train_featurization(cfg)
    days = list(chain.from_iterable([getattr(cfg.dataset, f"{split}_files") for split in splits]))
    sorted_paths = get_all_files_from_folders(cfg.transformation._graphs_dir, days)
    graph_list = [torch.load(path) for path in sorted_paths]

    words = []
    nodes = set()
    for G in log_tqdm(graph_list, desc="Get corpus with neighbors"):
        # Prepare the training data for Doc2Vec: each node and its neighbors as a 'document'
        for node in G.nodes():
            if node not in nodes:
                nodes.add(node)

                node_label = G.nodes[node]["label"]
                node_type = G.nodes[node]["node_type"]

                neighbors = list(G.neighbors(node))
                neighbor_labels = []

                for neighbor in neighbors:
                    label = G.nodes[neighbor]["label"]
                    type_ = G.nodes[neighbor]["node_type"]
                    neighbor_labels.extend(tokenize_label(label, type_))

                document = tokenize_label(node_label, node_type) + neighbor_labels

                if doc2vec_format:
                    words.append(TaggedDocument(words=document, tags=[node]))
                else:
                    words.append(document)

    return words
