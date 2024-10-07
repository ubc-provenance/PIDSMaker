import networkx as nx
import numpy as np
import json
import argparse
import logging
import os
import csv
import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import Counter
from collections import defaultdict
from gensim.models import Word2Vec
from unicodedata import category
import re
import torch
from config import *
from provnet_utils import *
import random
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import networkx as nx
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import networkx as nx

from provnet_utils import *
def read_graphs(split_files,cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    graph_list = []
    i = 0
    for path in tqdm(sorted_paths, desc='Building temporal random walks'):
        file = path.split("/")[-1]
        log(f"Process {path}")
        graph = torch.load(path)
        root_nodes = identify_root_nodes(graph)
        pseudo_graph = create_pseudo_graph(graph,root_nodes)
        new_G = prune_pseudo_roots(pseudo_graph, graph, 0.5)
        graph_list.append(new_G)
    documents,lens = generate_doc2vec_embeddings_for_graphs(graph_list, vector_size=128, epochs=10, model_path=cfg.featurization.embed_nodes.doc2vec._model_dir)
    output_file = './documents.txt'

    with open(output_file, 'w') as f:
        for doc in documents:
            tags = ' '.join(doc.tags)
            words = ' '.join(doc.words)
            f.write(f'Tags: [{tags}] Words: [{words}]\n')

    log(f'Documents saved to {output_file}')
    model_save_path = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(model_save_path,exist_ok=True)
    epochs = cfg.featurization.embed_nodes.doc2vec.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.doc2vec.alpha
    min_alpha = cfg.featurization.embed_nodes.doc2vec.min_alpha
    doc2vec(documents=documents,
                  model_save_path=model_save_path,
                  epochs=epochs,
                  emb_dim=emb_dim,
                  alpha=alpha,
                  min_alpha=min_alpha,
                  length=lens,
                  cfg=cfg)


def doc2vec(cfg,
            documents,
            model_save_path: str,
            epochs: int,
            emb_dim: int,
            alpha: float,
            min_alpha: float,
            length: int,
            dm: int = 1):
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    tagged_data = documents

    if use_seed:
        model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED)
    else:
        model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True)
    model.build_vocab(tagged_data)

    for epoch in range(epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=10, compute_loss=True)
        model.alpha -= 0.0002
        if model.alpha < min_alpha:
            model.alpha = min_alpha

        log(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')

    log(f'Saving Doc2Vec model to {model_save_path}')
    model.save(model_save_path + 'doc2vec_model.model')
    pass

def identify_root_nodes(G):
    root_nodes = set()

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        in_edges = list(G.in_edges(node, data=True))

        # out edge timestamp
        earliest_out_time = min(edge[2]['time'] for edge in out_edges) if out_edges else None
        # in
        earliest_in_time = min(edge[2]['time'] for edge in in_edges) if in_edges else None

        # if root
        if earliest_out_time is not None and (earliest_in_time is None or earliest_out_time < earliest_in_time):
            root_nodes.add(node)

    return root_nodes




def create_pseudo_graph(G,root_nodes):
    """
    Create a pseudo-graph G' based on the original graph G.
    Each pseudo-root retains the initial feature vector of the original root node,
    and outgoing edges are added from pseudo-roots to their descendants.

    Args:
        G (nx.DiGraph): Original directed graph with nodes and features.

    Returns:
        nx.DiGraph: Pseudo-graph with pseudo-root nodes and directed edges to descendants.
    """
    pseudo_graph = nx.DiGraph()

    # Step 1: Add all original nodes and edges to the pseudo-graph
    for node, attr in G.nodes(data=True):
        pseudo_graph.add_node(node, **attr)


    # Step 3: Create pseudo-root nodes and add edges to descendants
    for root in root_nodes:
        # Create pseudo-root node (retaining the same initial feature vector)
        pseudo_root = f"pseudo_{root}"
        pseudo_graph.add_node(pseudo_root, **G.nodes[root])  # Copy features from root node

        # Add edges from pseudo-root to all descendants of the original root
        descendants = nx.descendants(G, root)
        for descendant in descendants:
            pseudo_graph.add_edge(pseudo_root, descendant)

    return pseudo_graph

def prune_pseudo_roots(pseudo_graph, G, prune_threshold):
    """
    Prune pseudo-root nodes from the pseudo-graph if they connect to more than
    a certain percentage of nodes in the original provenance graph.

    Args:
        pseudo_graph (nx.DiGraph): The pseudo-graph with pseudo-root nodes.
        G (nx.DiGraph): The original provenance graph.
        prune_threshold (float): The threshold as a percentage (0-1) of total nodes in G.
                                 If a pseudo-root connects to more than this percentage of nodes,
                                 it will be pruned.

    Returns:
        nx.DiGraph: The pruned pseudo-graph.
    """
    total_nodes_in_G = len(G.nodes())
    max_allowed_connections = prune_threshold * total_nodes_in_G

    # Identify pseudo-roots that need to be pruned
    pseudo_roots_to_prune = []
    for node in pseudo_graph.nodes():
        if node.startswith("pseudo_"):
            # Count the number of nodes this pseudo-root connects to
            num_connections = len(list(pseudo_graph.successors(node)))
            if num_connections > max_allowed_connections:
                pseudo_roots_to_prune.append(node)

    # Prune the identified pseudo-root nodes from the pseudo-graph
    for pseudo_root in pseudo_roots_to_prune:
        pseudo_graph.remove_node(pseudo_root)

    return pseudo_graph


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os


def generate_doc2vec_embeddings_for_graphs(graph_list, vector_size=128, epochs=10, model_path='./models'):
    """
    Generate node embeddings for a list of graphs using Doc2Vec based on node labels and their neighbors.

    Args:
        graph_list (list of nx.Graph): List of input graphs.
        vector_size (int): The size of the resulting embeddings.
        epochs (int): Number of epochs for training Doc2Vec.
        model_path (str): Directory to save the trained Doc2Vec models.

    Returns:
        dict: A dictionary of node embeddings for each graph, keyed by graph index.
    """
    os.makedirs(model_path, exist_ok=True)  # Ensure the model directory exists

    all_embeddings = {}  # Store embeddings for all graphs
    documents = []
    nodes = []
    for idx, G in enumerate(graph_list):
        # Prepare the training data for Doc2Vec: each node and its neighbors as a 'document'
        for node in G.nodes():
            if node not in nodes:
                node_label = G.nodes[node].get('label', '')  # Assumes each node has a 'label' attribute
                parts = node_label.split(maxsplit=1)
                node_label = parts[1] if len(parts) > 1 else None
                neighbors = list(G.neighbors(node))
                neighbor_labels = []
                nodes.append(node)
                for neighbor in neighbors:
                    label = G.nodes[neighbor].get('label', '')
                    parts = label.split(maxsplit=1)
                    label = parts[1] if len(parts) > 1 else None
                    if isinstance(label, list):
                        label = ' '.join(label)
                    neighbor_labels.append(label)

                document = node_label + ' ' + ' '.join(neighbor_labels)
                documents.append(TaggedDocument(words=document.split(), tags=[str(node)]))


    return documents,len(nodes)


def main(cfg):
    use_seed = cfg.featurization.embed_nodes.use_seed

    if use_seed:
        SEED = 0
        np.random.seed(SEED)
        random.seed(SEED)

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    #os.makedirs(cfg.featurization.embed_nodes.word2vec._vec_graphs_dir, exist_ok=True)
    read_graphs(split_files=cfg.dataset.train_files,cfg = cfg)
    #read_graphs(split_files=cfg.dataset.val_files,cfg = cfg)
    #read_graphs(split_files=cfg.dataset.test_files,cfg = cfg)

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)