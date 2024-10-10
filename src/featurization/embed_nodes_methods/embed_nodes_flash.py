from provnet_utils import *
from config import *

from gensim.models import Word2Vec
import os
import torch
from tqdm import tqdm

def get_corpus(cfg):
    corpus = []

    data = load_graph_data(t='train', cfg=cfg)
    for d in data:
        corpus.append(d[0])
    return corpus

def load_graph_data(t, cfg):
    indexid2type, indexid2props = get_nid2props(cfg)

    if t == "train":
        split_files = cfg.dataset.train_files
    elif t == "test":
        split_files = cfg.dataset.test_files

    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)

    data_of_graphs = []

    for file_path in tqdm(sorted_paths, desc=f"Loading graph data of {t} set"):
        nx_g = torch.load(file_path)

        all_edges = []
        for u, v, key, attr in nx_g.edges(data=True, keys=True):
            edge = (u, v, attr['label'], int(attr['time']))
            all_edges.append(edge)
        sorted_edges = sorted(all_edges, key=lambda x: x[3])

        nodes, labels, edges = {}, {}, []
        for e in sorted_edges:
            src, dst, operation, t = e
            properties = [indexid2props[src] if indexid2props[src] is not None else []] + [operation] + [indexid2props[dst] if indexid2props[dst] is not None else []]

            if src not in nodes:
                nodes[src] = []
            if len(nodes[src]) < 300:
                nodes[src].extend(properties)
            labels[src] = indexid2type[src]

            if dst not in nodes:
                nodes[dst] = []
            if len(nodes[dst]) < 300:
                nodes[dst].extend(properties)
            labels[dst] = indexid2type[dst]

            edges.append((src, dst))

        features, feat_labels, edge_index, index_map = [], [], [[], []], {}
        for node_id, props in nodes.items():
            features.append(props)
            feat_labels.append(labels[node_id])
            index_map[node_id] = len(features) - 1

        #remove old function update_edge_index(edges, edge_index, index_map)
        for src_id, dst_id in edges:
            edge_index[0].append(index_map[src_id])
            edge_index[1].append(index_map[dst_id])

        data_of_graphs.append((features, feat_labels, edge_index, list(index_map.keys())))

    return data_of_graphs

def get_nid2props(cfg):
    cur, connect = init_database_connection(cfg)

    indexid2type = {}
    indexid2props = {}

    # netflow
    sql = """
            select dst_addr, index_id from netflow_node_table;
            """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        remote_ip = str(i[0])
        index_id = i[1]  # int

        indexid2type[str(index_id)] = ntype2id['netflow'] - 1 # 2
        indexid2props[str(index_id)] = remote_ip

    #subject
    sql = """
    select cmd, index_id from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        cmd = str(i[0])
        index_id = i[1]

        indexid2type[str(index_id)] = ntype2id['subject'] - 1 # 0
        indexid2props[str(index_id)] = cmd

    # file
    sql = """
       select path, index_id from file_node_table;
       """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        path = str(i[0])
        index_id = i[1]

        indexid2type[str(index_id)] = ntype2id['file'] - 1  # 1
        indexid2props[str(index_id)] = path

    return indexid2type, indexid2props

class RepeatableIterator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for phrases in self.data:
            for sentence in phrases:
                yield sentence

def main(cfg):
    log_start(__file__)
    model_save_dir = cfg.featurization.embed_nodes.flash._model_dir
    os.makedirs(model_save_dir, exist_ok=True)

    all_phrases = get_corpus(cfg=cfg)

    # Get hyper args
    vector_size = cfg.featurization.embed_nodes.flash.vector_size
    window = cfg.featurization.embed_nodes.flash.window
    min_count = cfg.featurization.embed_nodes.flash.min_count
    workers = cfg.featurization.embed_nodes.flash.workers
    epochs = cfg.featurization.embed_nodes.flash.epochs

    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

    model.build_vocab(RepeatableIterator(all_phrases), progress_per=10000)

    total_examples = model.corpus_count

    for epoch in range(epochs):
        log(f"Epoch #{epoch} start")
        model.train(RepeatableIterator(all_phrases), total_examples=total_examples, epochs=1)
        log(f"Epoch #{epoch} end")

    model.save(os.path.join(model_save_dir, "word2vec_model_final.model"))


if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)