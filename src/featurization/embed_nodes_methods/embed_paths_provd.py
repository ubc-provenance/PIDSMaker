from collections import defaultdict
from itertools import chain
from provnet_utils import *
from config import *
from featurization.featurization_utils import get_splits_to_train_featurization
from .embed_nodes_doc2vec import doc2vec

from sklearn.neighbors import LocalOutlierFactor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def get_edge_key(u, v, graph):
    edge_number_between_nodes = graph.new_edge_key(u, v) - 1
    edge_key = 0
    lowest_weight = 0
    for k in range(edge_number_between_nodes):
        if graph.edges[u, v, k]['weight'] <= lowest_weight:
            lowest_weight = graph.edges[u, v, k]['weight']
            edge_key = k
    return edge_key

def get_node2paths(G, k, max_path_length):
    def shortest_path(graph: nx.MultiDiGraph, src, max_path_length):
        if graph.out_degree(src) == 0:
            return None
        dist = dict.fromkeys(graph.nodes, float('inf'))
        dist[src] = 0
        path = dict.fromkeys(graph.nodes, None)
        topo_order = nx.topological_sort(graph)
        for n in topo_order:
            for s in graph.successors(n):
                # We are running the algorithm on -G.
                # So we should always select the edge with the lowest weight
                edge_key = get_edge_key(n, s, graph)
                if dist[s] > dist[n] + graph.edges[n, s, edge_key]['weight']:
                    dist[s] = dist[n] + graph.edges[n, s, edge_key]['weight']
                    path[s] = n

        # although we run the shortest path algorithm, the output is the longest path.
        dist[src] = float('inf')
        min_path_weight = min(dist.values())
        dst_nodes = [node for node in dist if dist[node] == min_path_weight]
        longest_paths = []
        for dst_node in dst_nodes:
            predecessor = path[dst_node]
            longest_path = [(predecessor, dst_node)]
            current_path_length = 1
            while predecessor != src:
                if current_path_length >= max_path_length:
                    break
                dst_node = predecessor
                predecessor = path[dst_node]
                longest_path.append((predecessor, dst_node))
                current_path_length += 1
            longest_paths.append([longest_path[::-1], min_path_weight])

        return longest_paths

    # reference: https://en.wikipedia.org/wiki/Longest_path_problem#Acyclic_graphs
    # According to the wiki, finding the longest path on a dag G is equal to find the shortest path on -G
    for (u, v, key) in G.edges(keys=True):
        G.edges[u, v, key]["weight"] = 0 - G.edges[u, v, key]["weight"]
    
    topo_order = nx.topological_sort(G)
    top_uncommon_node2paths = defaultdict(list)
    node2paths_cache = defaultdict(set) # we don't want duplicate paths for a node
    token_cache = {}
    def get_token(w):
        if w not in token_cache:
            token_cache[w] = tokenize_arbitrary_label(w)
        return token_cache[w]
    
    for src_node in topo_order:
        paths = shortest_path(G, src_node, max_path_length)
        if paths:
            for path in paths:
                path, score = path # todo: consider score + keep k
                path_str = str(path)
                if path_str not in node2paths_cache[src_node]:
                    node2paths_cache[src_node].add(path_str)
                    
                    text_path = text_extraction(path, G)
                    tokenized_path = list(chain(*[get_token(w) for w in text_path]))
                    
                    top_uncommon_node2paths[src_node].append([tokenized_path, score])

    return top_uncommon_node2paths

def text_extraction(path, G):
    text = []
    for edge in path:
        src = edge[0]
        dst = edge[1]

        src_text = G.nodes[src]['label']
        dst_text = G.nodes[dst]['label']

        # As in top-k path extraction algorithm, we always select the edge with the lowest weight.
        edge_key = get_edge_key(src, dst, G)
        edge_text = G.edges[src, dst, edge_key]['label']

        if not text:
            text.append(src_text)
            text.append(edge_text)
            text.append(dst_text)
        else:
            text.append(edge_text)
            text.append(dst_text)

    return text

def build_event_map(graph_list:list):
    event_mapping = {}
    num_of_host = len(graph_list)
    for graph in graph_list:
        for (u, v, k) in graph.edges:
            event = graph.nodes[u]["label"] + "->" + graph.edges[u, v, k]["label"] + "->" + graph.nodes[v]["label"]
            # Here, we assume each of 50 hosts has several graphs
            host_id = graph_list.index(graph) % num_of_host
            if event not in event_mapping:
                event_mapping[event] = [host_id]
            else:
                if host_id not in event_mapping[event]:
                    event_mapping[event].append(host_id)

    for event in event_mapping:
        count = len(event_mapping[event])
        event_mapping[event] = count
    return event_mapping

def weight_edge_list(glist, n, event_mapping):
    i = 0
    H = len(glist)  # the number of the logs represent the number of the host in the enterprise environment

    for G in glist:
        sorted_edges = sorted(G.edges(data=True, keys=True), key=lambda t: t[3].get('time'))

        min_time = sorted_edges[0][3]['time']
        max_time = sorted_edges[-1][3]['time']
        interval = (max_time - min_time) / n
        res_dic = {}

        pbar = log_tqdm(total=len(G.edges))
        pbar.set_description('load tasks：')
        for edge in sorted_edges:
            num_window = int((edge[3]['time'] - min_time) // interval)
            if edge[3]['time'] == max_time:
                num_window = n - 1
            if res_dic.get(edge[0]) is None:
                res_dic[edge[0]] = [[], []]  # [out], [in]
            if res_dic.get(edge[1]) is None:
                res_dic[edge[1]] = [[], []]  # 1 = true
            if num_window not in res_dic[edge[0]][0]:  # out & src node
                res_dic[edge[0]][0].append(num_window)
            if num_window not in res_dic[edge[1]][1]:  # in & dst node
                res_dic[edge[1]][1].append(num_window)
            pbar.update()

        for edge in log_tqdm(G.edges.data(keys=True), desc='weight edges of graph {}'.format(i)):
            # According to the paper, the weight of all pseudo_link is 1
            if G.edges[edge[0], edge[1], edge[2]]["label"] == "pseudo_link":
                G.edges[edge[0], edge[1], edge[2]]["Rareness"] = 0.5
                G.edges[edge[0], edge[1], edge[2]]["weight"] = 1
                G.edges[edge[0], edge[1], edge[2]]["in-val"] = 1
                G.edges[edge[0], edge[1], edge[2]]["out-val"] = 1
                continue

            src = G.nodes[edge[0]]
            dst = G.nodes[edge[1]]
            relation = G.edges[edge[0], edge[1], edge[2]]

            in_val = 1 - len(res_dic[edge[1]][1]) / n
            out_avl = 1 - len(res_dic[edge[0]][0]) / n
            G.edges[edge[0], edge[1], edge[2]]["in-val"] = in_val
            G.edges[edge[0], edge[1], edge[2]]["out-val"] = out_avl

            event = src["label"] + "->" + relation["label"] + "->" + dst["label"]
            h_e = event_mapping[event]
            r = out_avl * in_val * (h_e / H)

            if r > 0:
                G.edges[edge[0], edge[1], edge[2]]["Rareness"] = r
                G.edges[edge[0], edge[1], edge[2]]["weight"] = 0 - math.log(r, 2)
            else:
                with open('./weighted_error.txt', 'a+') as f:
                    f.write("Graph number [{}]: src: {}, dst: {}, in_val:{}, out_val:{}\n".format(
                        i, src, dst, in_val, out_avl))
                    f.close()
                if out_avl == 0:
                    out_avl = 1e-9
                if in_val == 0:
                    in_val = 1e-9
                r = out_avl * in_val * (h_e / H)
                G.edges[edge[0], edge[1], edge[2]]["Rareness"] = r
                G.edges[edge[0], edge[1], edge[2]]["weight"] = 0 - math.log(r, 2)
        i += 1
    return glist

def get_node2corpus(splits, cfg, model=None):
    n_time_windows = cfg.featurization.embed_nodes.provd.n_time_windows
    k = cfg.featurization.embed_nodes.provd.k
    mpl = cfg.featurization.embed_nodes.provd.mpl

    days = list(chain.from_iterable([getattr(cfg.dataset, f"{split}_files") for split in splits]))
    sorted_paths = get_all_files_from_folders(cfg.preprocessing.transformation._graphs_dir, days)
    train_g = [torch.load(path) for path in sorted_paths]
    
    # Weight graphs
    event_mapping = build_event_map(train_g)
    train_g = weight_edge_list(train_g, n_time_windows, event_mapping)

    # extract and embed paths
    train_files = sorted_paths
    corpus = []
    all_node2paths = defaultdict(list)
    for graph in log_tqdm(train_g, desc=f'Extracting the paths for {splits}'):
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            raise RuntimeError("The graph contains cycles, use transformation 'dag'.")

        node2paths = get_node2paths(graph, k, mpl)
        
        for node, paths in node2paths.items():
            all_node2paths[node].extend(paths)
            
    # For each node, we keep its k most uncommon paths
    for node, paths in all_node2paths.items():
        all_node2paths[node] = [path[0] for path in sorted(paths, key=(lambda x: x[1]))[:k]]
        
    return all_node2paths


def main(cfg):    
    model_save_dir = cfg.featurization.embed_nodes._model_dir

    splits = get_splits_to_train_featurization(cfg)
    node2corpus = get_node2corpus(splits, cfg)
    corpus = [list(chain(*v)) for k, v in node2corpus.items()]
    
    train_data = [TaggedDocument(text, tags=[corpus.index(text)]) for text in corpus]
    
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    epochs = cfg.featurization.embed_nodes.epochs
    alpha = cfg.featurization.embed_nodes.provd.alpha

    model = doc2vec(
        tagged_data=train_data,
        model_save_path=model_save_dir,
        epochs=epochs,
        emb_dim=emb_dim,
        alpha=alpha,
        cfg=cfg,
    )
    
    # Train classifier on training benign path embeddings
    path2vec = {}
    for _, corpus in node2corpus.items():
        for path in corpus:
            if str(path) not in path2vec:
                vector = model.infer_vector(path)
                path2vec[str(path)] = vector
    
    training_vectors = np.array(list(path2vec.values()))
    
    n_neighbors = cfg.featurization.embed_nodes.provd.n_neighbors
    contamination = cfg.featurization.embed_nodes.provd.contamination
    
    log("Training LOF model")
    clf = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors, contamination=contamination)
    clf.fit(training_vectors)
    torch.save(clf, os.path.join(model_save_dir, "lof.pkl"))


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
