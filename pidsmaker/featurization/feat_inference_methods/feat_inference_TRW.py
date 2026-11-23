import numpy as np
import torch
from gensim.models import Word2Vec

from pidsmaker.tokenization import load_tokenizer
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_indexid2msg,
    log_start,
    log_tqdm,
)


def cal_word_weight(n, percentage):
    d = -1 / n * percentage / 100
    a_1 = 1 / n - 0.5 * (n - 1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence


def main(cfg):
    log_start(__file__)
    base_dir = cfg.transformation._graphs_dir
    sorted_paths = get_all_graphs_for_dates(
        base_dir, cfg.dataset.train_dates + cfg.dataset.test_dates + cfg.dataset.val_dates
    )
    used_nodes = set()
    for file_path in log_tqdm(sorted_paths, desc="Get nodes in graphs"):
        graph = torch.load(file_path)
        used_nodes = used_nodes | set(graph.nodes())
    used_nodes = list(used_nodes)

    indexid2msg = get_indexid2msg(cfg)

    trw_word2vec_model_path = cfg.featurization._model_dir + "trw_word2vec.model"
    model = Word2Vec.load(trw_word2vec_model_path)
    decline_percentage = cfg.featurization.temporal_rw.decline_rate

    tokenizer = load_tokenizer(cfg)
    indexid2vec = {}
    label2vec = {}  # the embedding only depends on (node_type, label)
    for indexid in log_tqdm(used_nodes, desc="Embeding all nodes in the dataset"):
        msg = indexid2msg[indexid]
        node_type, node_label = msg[0], msg[1]
        cached = label2vec.get((node_type, node_label))
        if cached is not None:
            indexid2vec[indexid] = cached
            continue
        tokens = tokenizer.tokenize(node_label, node_type)
        if not tokens:
            indexid2vec[indexid] = label2vec[(node_type, node_label)] = np.zeros(model.vector_size)
            continue

        weight_list = cal_word_weight(len(tokens), decline_percentage)
        word_vectors = [model.wv[word] for word in tokens]
        weighted_vectors = [
            weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)
        ]
        sentence_vector = np.mean(weighted_vectors, axis=0)

        normalized_vector = sentence_vector / (np.linalg.norm(sentence_vector) + 1e-12)
        indexid2vec[indexid] = label2vec[(node_type, node_label)] = np.array(normalized_vector)

    return indexid2vec
