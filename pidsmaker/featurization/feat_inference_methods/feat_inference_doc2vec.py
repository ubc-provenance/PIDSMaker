import os
import zlib

import numpy as np
from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.featurization_methods.featurization_doc2vec import doc2vec_label_fn
from pidsmaker.tokenization import load_tokenizer
from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm


def infer_node_vector(model, tokens):
    """Normalized doc2vec vector of a node's tokens, the same whatever the nodes inferred before.

    `infer_vector` draws its negative samples from the model's shared random state, which is
    seeded here from the tokens. A node whose tokens have no characters (e.g. a process
    without path, whose name is removed by `truncate_process_name`: its label "" gives the
    tokens [""]) gets a zero vector: gensim seeds its initial vector from the joined tokens,
    and starts it from an unseeded random vector when they are empty, different on every run.
    """
    if not "".join(tokens):
        return np.zeros(model.dv.vector_size, dtype=np.float32)
    model.random.seed(zlib.crc32(" ".join(tokens).encode()))
    vector = model.infer_vector(tokens)
    return vector / np.linalg.norm(vector)


def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    doc2vec_model_path = os.path.join(cfg.featurization._model_dir, "doc2vec_model.model")
    model = Doc2Vec.load(doc2vec_model_path)

    tokenizer = load_tokenizer(cfg)
    label_fn = doc2vec_label_fn(cfg)
    indexid2vec = {}
    # Nodes with the same tokens share their vector (inferred once)
    tokens2vec = {}
    sorted_indexid2msg = dict(sorted(indexid2msg.items(), key=lambda item: int(item[0])))
    for indexid, msg in log_tqdm(sorted_indexid2msg.items(), desc="Embedding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        if label_fn is not None:
            node_label = label_fn(node_type, node_label)
        tokens = tokenizer.tokenize(node_label, node_type)

        key = tuple(tokens)
        normalized_vector = tokens2vec.get(key)
        if normalized_vector is None:
            normalized_vector = infer_node_vector(model, tokens)
            tokens2vec[key] = normalized_vector
        indexid2vec[indexid] = normalized_vector

    return indexid2vec
