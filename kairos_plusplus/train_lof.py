from sklearn.neighbors import LocalOutlierFactor
import argparse

from config import *
from provnet_utils  import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-node2vec_train_val_path',
                        help='Input the path to stored node2vec (train and val datasets only) maps from Word2Vec.',
                        required=False)
    parser.add_argument('-lof_path', help='The path to save lof model', required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    node2vec_train_val_path = args.node2vec_train_val_path
    lof_path = args.lof_path

    labels_and_embeddings = torch.load(node2vec_train_val_path)

    embeddings = []
    for label in labels_and_embeddings:
        # We don't consider network
        if "." and ":" in label:
            continue
        else:
            embeddings.append(labels_and_embeddings[label])

    clf = LocalOutlierFactor(novelty=True, n_neighbors=30).fit(embeddings)
    torch.save(clf, lof_path)



