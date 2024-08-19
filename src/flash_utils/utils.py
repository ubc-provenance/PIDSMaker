from provnet_utils import *
from config import *
import torch
from tqdm import tqdm

def load_one_graph_data(graph_path, indexid2type, indexid2props):
    nx_g = torch.load(graph_path)

    all_edges = []
    for u, v, key, attr in nx_g.edges(data=True, keys=True):
        edge = (u, v, attr['label'], int(attr['time']))
        all_edges.append(edge)
    sorted_edges = sorted(all_edges, key=lambda x: x[3])

    nodes, labels, edges = {}, {}, []
    for e in sorted_edges:
        src, dst, operation, t = e
        properties = [indexid2props[src] if indexid2props[src] is not None else []] + [operation] + [
            indexid2props[dst] if indexid2props[dst] is not None else []]

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

    update_edge_index(edges, edge_index, index_map)

    return features, feat_labels, edge_index, list(index_map.keys())


def load_graph_data(t, cfg):
    indexid2type, indexid2props = get_nid2props(cfg)

    if t == "train":
        split_files = cfg.dataset.train_files
    elif t == "test":
        split_files = cfg.dataset.test_files

    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
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

        update_edge_index(edges, edge_index, index_map)

        data_of_graphs.append((features, feat_labels, edge_index, list(index_map.keys())))

    return data_of_graphs



def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

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
