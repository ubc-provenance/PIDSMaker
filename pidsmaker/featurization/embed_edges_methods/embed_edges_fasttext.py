import os

import numpy as np
from gensim.models import FastText

from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm, tokenize_label


def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    model_path = os.path.join(cfg.featurization.embed_nodes._model_dir, "fasttext.pkl")
    model = FastText.load(model_path)

    indexid2vec = {}
    for indexid, msg in log_tqdm(indexid2msg.items(), desc="Embeding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        tokens = tokenize_label(node_label, node_type)

        word_vectors = [model.wv[word] for word in tokens]
        sentence_vector = np.mean(word_vectors, axis=0)

        normalized_vector = sentence_vector / (np.linalg.norm(sentence_vector) + 1e-12)
        indexid2vec[indexid] = np.array(normalized_vector)

    return indexid2vec
