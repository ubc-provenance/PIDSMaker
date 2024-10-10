from provnet_utils import *
from config import *

from gensim.models.doc2vec import Doc2Vec

def get_edge_key(u, v, graph):
    edge_number_between_nodes = graph.new_edge_key(u, v) - 1
    edge_key = 0
    lowest_weight = 0
    for k in range(edge_number_between_nodes):
        if graph.edges[u, v, k]['weight'] <= lowest_weight:
            lowest_weight = graph.edges[u, v, k]['weight']
            edge_key = k
    return edge_key

def top_k_path(G, k, max_path_length):
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
    top_uncommon_paths = []
    topo_order = nx.topological_sort(G)
    for src_node in topo_order:
        paths = shortest_path(G, src_node, max_path_length)
        if paths:
            for path in paths:
                if path not in top_uncommon_paths:
                    top_uncommon_paths.append(path)

    top_uncommon_paths = sorted(top_uncommon_paths, key=(lambda x: x[1]))[:k]
    return top_uncommon_paths

def text_extraction(path, G):
    text = []
    for edge in path:
        src = edge[0]
        dst = edge[1]

        src_text = G.nodes[src]['name'] if "name" in list(dict(G.nodes[src]).keys()) else G.nodes[src]['label']
        dst_text = G.nodes[dst]['name'] if "name" in list(dict(G.nodes[dst]).keys()) else G.nodes[dst]['label']

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

def build_text_list(path_list, G):
    text_list = []
    for path in path_list:
        text_list.append(
            text_extraction(path[0], G)
        )
    return text_list

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

        pbar = tqdm(total=len(G.edges))
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

        for edge in tqdm(G.edges.data(keys=True), desc='weight edges of graph {}'.format(i)):
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

def path_embedding(text_list, model_path, alpha, dm=1, vector_size=100, epochs=100, window_size=5, is_train=True):
    if is_train:
        train_data = []
        for text in text_list:
            document = TaggedDocument(text, tags=[text_list.index(text)])
            train_data.append(document)

        model = Doc2Vec(train_data, dm=dm, vector_size=vector_size, window=window_size, alpha=alpha, seed=0)
        model.train(train_data, epochs=epochs, total_examples=model.corpus_count)
        model.save(model_path)
    else:
        model = Doc2Vec.load(model_path)

    path_vectors = []
    for text in text_list:
        model.random.seed(0)
        pv = model.infer_vector(text)
        path_vectors.append(pv)

    return path_vectors

def extract_and_embed_paths(split_files, paras, cfg,type_data,is_test,trial=None, is_finetune = False):
    n_time_windows = paras["n_time_windows"]
    k = paras["k"]
    vd = paras["vd"]
    ws = paras["ws"]
    epoch = paras["epoch"]
    alpha = paras["alpha"]
    mpl = paras["mpl"]

    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    train_g = [torch.load(path) for path in sorted_paths]
    # Weight graphs
    event_mapping = build_event_map(train_g)
    train_g = weight_edge_list(train_g, n_time_windows, event_mapping)


    output_dir = cfg.featurization.embed_edges._edge_embeds_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not is_test:
        # extract and embed paths
        train_files = sorted_paths
        text_list = []
        training_info = []
        i = 0
        for graph in tqdm(train_g, desc='Extracting the paths from training set'):
            cycles = list(nx.simple_cycles(graph))

            if cycles:
                print("The graph contains cycles.")
            else:
                print("The graph does not contain cycles.")
            top_uncommon_paths = top_k_path(graph, k, mpl)
            texts = build_text_list(top_uncommon_paths, graph)
            text_list += texts
            training_info.append([train_files[i], texts])
            i += 1
        train_pv = path_embedding(
            text_list=text_list,
            model_path=output_dir +type_data,
            alpha=alpha,
            vector_size=vd,
            epochs=epoch,
            window_size=ws,
            is_train=True)

        train_pv_path = output_dir+type_data + "embeddings.npy"
        np.save(train_pv_path, train_pv)

    if is_test:
        tested_graphs = {}
        test_files = sorted_paths
        i = 0
        for file in test_files:
            tested_graphs[file] = train_g[i]
            i += 1
        tested_paths = {}
        for graph in tqdm(train_g, desc='Extracting the paths from testing set'):
            top_uncommon_paths = top_k_path(graph, k, mpl)
            text_list = build_text_list(top_uncommon_paths, graph)
            path_vectors = path_embedding(
                text_list=text_list,
                model_path=output_dir +type_data,
                alpha=alpha,
                vector_size=vd,
                epochs=epoch,
                window_size=ws,
                is_train=False)
            tested_paths[file_name] = [path_vectors, top_uncommon_paths, text_list]
        test_pv_path = output_dir +type_data,+type_data + "embeddings.npy"
        with open(test_pv_path, "wb") as f:
            pickle.dump(tested_paths, f)
            f.close()
        #test_path_stats_path = "./Test_path" + "test_path_stats.csv"
    #report_paths_stats_info(tested_paths, test_path_stats_path,k)
    return 0

def main(cfg):
    paras = {}
    paras["n_time_windows"] = 100
    paras["k"] = 20
    paras["vd"] = 100
    paras["ws"] = 5
    paras["epoch"] = 100
    paras["alpha"] = 0.025
    paras["n_neighbors"] = 20
    paras["contamination"] = 0.04
    paras["mpl"] = 10

    extract_and_embed_paths(cfg.dataset.train_files,  paras,cfg,"train",False,trial=None, is_finetune=False)
    #extract_and_embed_paths(cfg.dataset.val_files, paras,cfg, "val",False, trial=None, is_finetune=False)
    extract_and_embed_paths(cfg.dataset.test_files,  paras, cfg,"test",True, trial=None, is_finetune=False)


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
