import os

import numpy as np
from gensim.models.doc2vec import Doc2Vec

from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm, tokenize_label


def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    doc2vec_model_path = os.path.join(
        cfg.featurization._model_dir, "doc2vec_model.model"
    )
    model = Doc2Vec.load(doc2vec_model_path)

    indexid2vec = {}
    for indexid, msg in log_tqdm(indexid2msg.items(), desc="Embeding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        tokens = tokenize_label(node_label, node_type)

        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)
        indexid2vec[indexid] = normalized_vector

    return indexid2vec
