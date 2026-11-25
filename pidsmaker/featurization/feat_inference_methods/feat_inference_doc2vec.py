import os

import numpy as np
from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.featurization_methods.featurization_doc2vec import doc2vec_label_fn
from pidsmaker.tokenization import load_tokenizer
from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm


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
            vector = model.infer_vector(tokens)
            normalized_vector = vector / np.linalg.norm(vector)
            tokens2vec[key] = normalized_vector
        indexid2vec[indexid] = normalized_vector

    return indexid2vec
