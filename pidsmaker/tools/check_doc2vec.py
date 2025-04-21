import tqdm
import os
from gensim.models.doc2vec import Doc2Vec
from config import get_runtime_required_args, get_yml_cfg
from provnet_utils import log, tokenize_subject, tokenize_file, tokenize_netflow, log_start, get_indexid2msg
import pandas

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def get_msg2vec(indexid2msg, model_path):

    model = Doc2Vec.load(model_path)
    log(f"Loaded model from {model_path}")

    num_sub, num_file, num_net = 0, 0, 0
    msg2vec = {}
    for indexid, msg in tqdm(indexid2msg.items(), desc='processing indexid2vec:'):
        if msg[0] == 'subject':
            tokens = tokenize_subject(msg[1])
            label = 0
            num_sub += 1
        if msg[0] == 'file':
            tokens = tokenize_file(msg[1])
            label = 1
            num_file += 1
        if msg[0] == 'netflow':
            tokens = tokenize_netflow(msg[1])
            label = 2
            num_net += 1

        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)

        msg2vec[msg[1]] = [label, normalized_vector]

    log(f"Number of subjects: {num_sub}; Number of files: {num_file}; Number of netflows: {num_net}")

    log(f"Finish generating normalized node vectors.")

    return msg2vec

def get_attrs_different_files(indexid2msg, model_path):
    model = Doc2Vec.load(model_path)
    log(f"Loaded model from {model_path}")

    embeddings = {}

    file_list_1 = []
    for i in range(1516491, 1516491 + 200):
        tokens = tokenize_file(indexid2msg[i][1])
        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)
        file_list_1.append(normalized_vector)
    embeddings['/data/*'] = {
        'embeddings': file_list_1,
        'color': 'r'
    }

    file_list_2 = []
    for i in range(1520698,1520897+1):
        tokens = tokenize_file(indexid2msg[i][1])
        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)
        file_list_2.append(normalized_vector)
    embeddings['/proc/*'] = {
        'embeddings': file_list_2,
        'color': 'g'
    }

    file_list_3 = []
    for i in list(range(1557820,1557855+1)) + list(range(1567860, 1567881+1)):
        tokens = tokenize_file(indexid2msg[i][1])
        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)
        file_list_3.append(normalized_vector)
    embeddings['/home/admin/*'] = {
        'embeddings': file_list_3,
        'color': 'b'
    }

    file_list_4 = []
    for i in range(1420793,1421042):
        tokens = tokenize_file(indexid2msg[i][1])
        vector = model.infer_vector(tokens)
        normalized_vector = vector / np.linalg.norm(vector)
        file_list_4.append(normalized_vector)
    embeddings['/usr/share/*'] = {
        'embeddings': file_list_4,
        'color': 'yellow'
    }

    return embeddings

def get_attrs_one_type(msg2vec, num_each_type, sampled_type):
    embeddings_sub = []

    for key,value in msg2vec.items():
        if value[0] == sampled_type:
            embeddings_sub.append(value[1])

    if sampled_type == 0:
        type_name = 'subject'
        color = 'r'
    elif sampled_type == 1:
        type_name = 'file'
        color = 'g'
    elif sampled_type == 2:
        type_name = 'netflow'
        color = 'b'

    embeddings = {}
    sampled_sub = random.sample(range(len(embeddings_sub)), num_each_type)
    embeddings[type_name] = {
        'embeddings': [embeddings_sub[i] for i in sampled_sub],
        'color': color
    }

    log(f"Finish generating node embeddings.")

    return embeddings

def get_attrs_each_type(msg2vec, num_each_type):

    embeddings_sub = []
    embeddings_file = []
    embeddings_net = []

    for key,value in msg2vec.items():
        if value[0] == 0:
            embeddings_sub.append(value[1])
        elif value[0] == 1:
            embeddings_file.append(value[1])
        elif value[0] == 2:
            embeddings_net.append(value[1])

    log(f"Number fo subjects: {len(embeddings_sub)}")
    log(f"Number of files: {len(embeddings_file)}")
    log(f"Number of netflows: {len(embeddings_net)}")

    embeddings = {}
    sampled_sub = random.sample(range(len(embeddings_sub)), num_each_type)
    embeddings['subject'] = {
        'embeddings': [embeddings_sub[i] for i in sampled_sub],
        'color': 'r'
    }

    sampled_file = random.sample(range(len(embeddings_file)), num_each_type)
    embeddings['file'] = {
        'embeddings': [embeddings_file[i] for i in sampled_file],
        'color': 'g'
    }

    sampled_net = random.sample(range(len(embeddings_net)), num_each_type)
    embeddings['netflow'] = {
        'embeddings': [embeddings_net[i] for i in sampled_net],
        'color': 'b'
    }

    log(f"Finish generating node embeddings.")

    return embeddings

def get_attrs(msg2vec):
    element_info = []
    embeddings = []

    for key,value in msg2vec.items():
        element_info.append([key, value[0]])
        embeddings.append(value[1])

    log(f"Finish generating node embeddings.")

    return element_info, embeddings


def get_vec_csv(msg2vec, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    element_info_path = os.path.join(csv_path, 'element_info.csv')
    embedding_path = os.path.join(csv_path, 'embeddings.csv')

    element_info = []
    embeddings = []

    for key,value in msg2vec.items():
        element_info.append([key, value[0]])
        embeddings.append(value[1])

    element_info_df = pandas.DataFrame(element_info)
    embeddings_df = pandas.DataFrame(embeddings)
    element_info_df.to_csv(element_info_path, index=False, header=False)
    log(f"Saving element info to {element_info_path}")
    embeddings_df.to_csv(embedding_path, index=False, header=False)
    log(f"Saving embeddings to: {embedding_path}")
    pass

def build_tsne_visualization(embeddings, fig_path):
    log(f"Building TSNE visualization")
    log("Building TSNE visualization")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100, metric='euclidean', init='pca')

    dic_2d = {}

    for key, value in embeddings.items():
        doc2vec = np.stack(value['embeddings'])
        embeddings_2d = tsne.fit_transform(doc2vec)

        dic_2d[key] ={
            "embeddings_2d": embeddings_2d,
            "color": value['color']
        }

    log(f"Start visualizing TSNE embeddings")
    plt.figure(figsize=(10, 8))

    for key,value in dic_2d.items():
        plt.scatter(value['embeddings_2d'][:, 0], value['embeddings_2d'][:, 1], marker='.', color=value['color'], label=key)

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(fig_path)
    log(f"Fig saved to {fig_path}")

def main(cfg):
    log_start(__file__)
    indexid2msg = get_indexid2msg(cfg)

    doc2vec_model_path = cfg.featurization.embed_nodes._model_dir + 'doc2vec_model.model'
    msg2vec = get_msg2vec(indexid2msg, doc2vec_model_path)

    # get_vec_csv(msg2vec, cfg.featurization.embed_nodes._task_path)


    fig_name = f"tsne_different_files.png"
    fig_path = os.path.join(cfg.featurization.embed_nodes._task_path, fig_name)

    # embeddings = get_attrs_each_type(msg2vec, num_each_type=1000)
    # embeddings = get_attrs_one_type(msg2vec, num_each_type=1500, sampled_type=2)
    embeddings = get_attrs_different_files(indexid2msg, doc2vec_model_path)

    build_tsne_visualization(embeddings, fig_path)



if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)