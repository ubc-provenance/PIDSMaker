import numpy as np
from sklearn.feature_extraction import FeatureHasher

from pidsmaker.utils.utils import get_indexid2msg, log_start, log_tqdm


def path2higlist(p):
    l = []
    spl = p.strip().split("/")
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + "/" + i)
        else:
            l.append(i)
    return l


def ip2higlist(p):
    l = []
    spl = p.strip().split(".")
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + "." + i)
        else:
            l.append(i)
    return l


def list2str(l):
    s = ""
    for i in l:
        s += i
    return s


def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    emb_dim = cfg.feat_training.emb_dim
    FH_string = FeatureHasher(n_features=emb_dim, input_type="string")

    indexid2vec = {}
    for indexid, msg in log_tqdm(indexid2msg.items(), desc="Embeding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        if node_type == "subject" or node_type == "file":
            higlist = path2higlist(node_label)
        else:
            higlist = ip2higlist(node_label)
        higstr = list2str(higlist)

        dense_vector = FH_string.fit_transform([higstr]).toarray()

        normalized_vector = dense_vector / (np.linalg.norm(dense_vector) + 1e-12)
        indexid2vec[indexid] = normalized_vector.squeeze()

    return indexid2vec
