import os
from collections import defaultdict
from itertools import chain

import torch
from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.provnet_utils import (
    get_all_files_from_folders,
    get_indexid2msg,
    log,
    log_start,
    log_tqdm,
    tokenize_arbitrary_label,
)


def get_node2corpus(cfg, splits):
    indexid2msg = get_indexid2msg(cfg)

    days = list(chain.from_iterable([getattr(cfg.dataset, f"{split}_files") for split in splits]))
    sorted_paths = get_all_files_from_folders(cfg.preprocessing.transformation._graphs_dir, days)

    data_of_graphs = []

    for file_path in log_tqdm(sorted_paths, desc=f"Loading training data for {str(splits)}"):
        graph = torch.load(file_path)

        sorted_edges = sorted(
            [
                (u, v, attr["label"], int(attr["time"]))
                for u, v, key, attr in graph.edges(data=True, keys=True)
            ],
            key=lambda x: x[3],
        )

        nodes, node_types, edges = defaultdict(list), {}, []
        for e in sorted_edges:
            src, dst, operation, t = e
            src_type, src_msg = indexid2msg[src]
            dst_type, dst_msg = indexid2msg[dst]

            properties = [src_msg, operation, dst_msg]

            if len(nodes[src]) < 300:
                nodes[src].extend(properties)
            node_types[src] = src_type

            if len(nodes[dst]) < 300:
                nodes[dst].extend(properties)
            node_types[dst] = dst_type

            edges.append((src, dst))

        features, types, index_map = [], [], {}
        for node_id, props in nodes.items():
            features.append(props)
            types.append(node_types[node_id])
            index_map[node_id] = len(features) - 1

        data_of_graphs.append((features, types, list(index_map.keys())))

    token_cache = {}
    node2corpus = defaultdict(list)
    for graphs in log_tqdm(data_of_graphs, desc="Tokenizing corpus"):
        for msg, node_type, node_id in zip(graphs[0], graphs[1], graphs[2]):
            for sentence in msg:
                if sentence not in token_cache:
                    tokens = tokenize_arbitrary_label(sentence)
                    token_cache[sentence] = tokens

                tokens = token_cache[sentence]
                node2corpus[node_id].extend(tokens)

    return node2corpus


class RepeatableIterator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for phrases in self.data:
            for sentence in phrases:
                yield sentence


def main(cfg):
    log_start(__file__)

    training_files = get_splits_to_train_featurization(cfg)
    all_phrases = list(get_node2corpus(cfg=cfg, splits=training_files).values())

    emb_dim = cfg.featurization.embed_nodes.emb_dim
    epochs = cfg.featurization.embed_nodes.epochs
    min_count = cfg.featurization.embed_nodes.flash.min_count
    workers = cfg.featurization.embed_nodes.flash.workers

    log("Training word2vec model...")
    model = Word2Vec(vector_size=emb_dim, min_count=min_count, workers=workers, epochs=epochs)
    model.build_vocab(RepeatableIterator(all_phrases), progress_per=10000)
    model.train(RepeatableIterator(all_phrases), total_examples=model.corpus_count, epochs=epochs)

    model_save_dir = cfg.featurization.embed_nodes._model_dir
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(os.path.join(model_save_dir, "word2vec_model_final.model"))
