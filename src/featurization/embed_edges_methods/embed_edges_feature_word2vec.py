from provnet_utils import *
from config import *

from gensim.models import Word2Vec


def cal_word_weight(n, percentage):
    d = -1 / n * percentage / 100
    a_1 = 1/n - 0.5 * (n-1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence

def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    feature_word2vec_model_path = cfg.featurization.embed_nodes._model_dir + 'feature_word2vec.model'
    model = Word2Vec.load(feature_word2vec_model_path)
    
    decline_percentage = cfg.featurization.embed_nodes.feature_word2vec.decline_rate

    zeros = np.zeros((cfg.featurization.embed_nodes.emb_dim,))
    indexid2vec = {}
    for indexid, msg in log_tqdm(indexid2msg.items(), desc='Embeding all nodes in the dataset'):
        node_type, node_label = msg[0], msg[1]
        tokens = tokenize_label(node_label, node_type)

        weight_list = cal_word_weight(len(tokens), decline_percentage)

        word_vectors = [model.wv[word] if word in model.wv else zeros for word in tokens]
        weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
        sentence_vector = np.mean(weighted_vectors, axis=0)

        normalized_vector = sentence_vector / np.linalg.norm(sentence_vector) + 1e-12
        indexid2vec[indexid] = np.array(normalized_vector)

    return indexid2vec
