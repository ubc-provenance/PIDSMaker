import numpy as np
from sklearn.feature_extraction import FeatureHasher

from pidsmaker.tokenization import load_tokenizer
from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm


def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    emb_dim = cfg.featurization.emb_dim
    FH_string = FeatureHasher(n_features=emb_dim, input_type="string")

    tokenizer = load_tokenizer(cfg)
    indexid2vec = {}
    sorted_indexid2msg = dict(sorted(indexid2msg.items(), key=lambda item: int(item[0])))
    label2vec = {}  # the embedding only depends on (node_type, label)
    for indexid, msg in log_tqdm(sorted_indexid2msg.items(), desc="Embedding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        cached = label2vec.get((node_type, node_label))
        if cached is not None:
            indexid2vec[indexid] = cached
            continue
        tokens = tokenizer.tokenize(node_label, node_type)
        dense_vector = FH_string.fit_transform([tokens]).toarray()

        normalized_vector = dense_vector / (np.linalg.norm(dense_vector) + 1e-12)
        indexid2vec[indexid] = label2vec[(node_type, node_label)] = normalized_vector.squeeze()

    return indexid2vec
