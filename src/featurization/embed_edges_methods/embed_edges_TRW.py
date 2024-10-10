from provnet_utils import *
from config import *

from gensim.models import Word2Vec


def cal_word_weight(n,percentage):
    d = -1 / n * percentage / 100
    a_1 = 1/n - 0.5 * (n-1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence

def main(cfg):
    log_start(__file__)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, (cfg.dataset.train_files +
                                                         cfg.dataset.test_files +
                                                         cfg.dataset.val_files))
    used_nodes = set()
    for file_path in tqdm(sorted_paths, desc="Get nodes in graphs"):
        graph = torch.load(file_path)
        used_nodes = used_nodes | set(graph.nodes())
    used_nodes = list(used_nodes)
    
    indexid2msg = get_indexid2msg(cfg)
    
    trw_word2vec_model_path = cfg.featurization.embed_nodes.temporal_rw._model_dir + 'trw_word2vec.model'
    model = Word2Vec.load(trw_word2vec_model_path)
    decline_percentage = cfg.featurization.embed_nodes.temporal_rw.decline_rate

    indexid2vec = {}
    for indexid in tqdm(used_nodes, desc='Embeding all nodes in the dataset'):
        msg = indexid2msg[indexid]
        node_type, node_label = msg[0], msg[1]
        tokens = tokenize_label(node_label, node_type)

        weight_list = cal_word_weight(len(tokens), decline_percentage)
        word_vectors = [model.wv[word] for word in tokens]
        weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
        sentence_vector = np.mean(weighted_vectors, axis=0)

        normalized_vector = sentence_vector / np.linalg.norm(sentence_vector)
        indexid2vec[indexid] = np.array(normalized_vector)

    return indexid2vec
