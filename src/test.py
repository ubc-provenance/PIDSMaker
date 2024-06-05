import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec

def get_indexid2vec(indexid2msg, model_path, use_node_types, decline_percentage, node_list):
    model = Word2Vec.load(model_path)
    print(f"Loaded model from {model_path}")

    indexid2vec = {}
    for indexid in tqdm(node_list, desc='processing indexid2vec:'):
        msg = indexid2msg[int(indexid)]
        if msg[0] == 'subject':
            if use_node_types:
                tokens = tokenize_subject(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_subject(msg[1])
        elif msg[0] == 'file':
            if use_node_types:
                tokens = tokenize_file(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_file(msg[1])
        else:
            if use_node_types:
                tokens = tokenize_netflow(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_netflow(msg[1])

        weight_list = cal_word_weight(len(tokens), decline_percentage)
        word_vectors = [model.wv[word] for word in tokens]
        weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
        sentence_vector = np.mean(weighted_vectors, axis=0)

        normalized_vector = sentence_vector / np.linalg.norm(sentence_vector)

        indexid2vec[int(indexid)] = np.array(normalized_vector)

    print(f"Finish generating normalized node vectors.")

    return indexid2vec

def cal_word_weight(n,percentage):
    d = -1 / n * percentage / 100
    a_1 = 1/n - 0.5 * (n-1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence

def main(cfg):
    use_node_types = cfg.featurization.embed_nodes.temporal_rw.use_node_types
    use_cmd = cfg.featurization.embed_nodes.temporal_rw.use_cmd
    use_port = cfg.featurization.embed_nodes.temporal_rw.use_port
    decline_rate = cfg.featurization.embed_nodes.temporal_rw.decline_rate

    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, (cfg.dataset.train_files +
                                                         cfg.dataset.test_files +
                                                         cfg.dataset.val_files))
    used_nodes = set()
    for file_path in tqdm(sorted_paths, desc="get nodes in graphs:"):
        graph = torch.load(file_path)
        used_nodes = used_nodes | set(graph.nodes())
    used_nodes = list(used_nodes)

    print("Loading node msg from database...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)

    print("Generating node vectors...")
    trw_word2vec_model_path = cfg.featurization.embed_nodes.temporal_rw._model_dir + 'trw_word2vec.model'
    indexid2vec = get_indexid2vec(indexid2msg=indexid2msg, model_path=trw_word2vec_model_path,
                                  use_node_types=use_node_types, decline_percentage=decline_rate,
                                  node_list=used_nodes)




if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)