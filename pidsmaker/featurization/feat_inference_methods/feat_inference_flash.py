import math
import os

import numpy as np
import torch
from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_methods.featurization_flash import get_node2corpus
from pidsmaker.utils.utils import log_start, log_tqdm


def infer(document, w2vmodel, encoder):
    """
    Each node is associated to a `document` which is the list of (msg => edge type => msg)
    involving this node.
    We get the embedding of each word inside this document and we do the mean of all embeddings.
    OOV words are simply ignored.
    """
    # Same values as stacking `w2vmodel.wv[word]` for the words in `w2vmodel.wv` (the tokens are
    # strings, so `word in wv` is a lookup in `key_to_index`), gathered at once
    key_to_index = w2vmodel.wv.key_to_index
    indices = [key_to_index[word] for word in document if word in key_to_index]

    embedding_dim = w2vmodel.vector_size

    if not indices:
        return np.zeros(embedding_dim)

    word_embeddings_array = w2vmodel.wv.vectors[indices]

    # float32 additions, as with the torch tensors before
    if len(document) < 100000:
        word_embeddings_array = encoder.embed_numpy(word_embeddings_array)

    return np.mean(word_embeddings_array, axis=0)


class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe_numpy = self.pe.numpy()

    def embed(self, x):
        return x + self.pe[: x.size(0)]

    def embed_numpy(self, x):
        """`embed` on a float32 numpy array."""
        return x + self.pe_numpy[: x.shape[0]]


def main(cfg):
    log_start(__file__)

    trained_w2v_dir = cfg.featurization._model_dir
    w2vmodel = Word2Vec.load(os.path.join(trained_w2v_dir, "word2vec_model_final.model"))
    w2v_vector_size = cfg.featurization.emb_dim

    node2corpus = get_node2corpus(cfg, splits=["train", "val", "test"])
    indexid2vec = {}
    # The encoder is a constant (max_len, d) table: build it once rather than once per node
    encoder = PositionalEncoder(w2v_vector_size)
    for indexid, corpus in log_tqdm(node2corpus.items(), desc="Embeding all nodes in the dataset"):
        indexid2vec[indexid] = infer(corpus, w2vmodel, encoder)

    return indexid2vec
