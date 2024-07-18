from config import *
from provnet_utils import *
import wandb
import os
import torch
import networkx as nx

def convert_DAG(graph):
    print(graph)

    edges = []
    node_version = {}
    for u, v, k, data in graph.edges(keys=True, data=True):
        edges.append((u, v, int(data['time'])))
        if u not in node_version:
            node_version[u] = 0
        if v not in node_version:
            node_version[v] = 0

    sorted_edges = sorted(edges, key=lambda x: x[2])

    new_nodes = set()
    new_edges = []
    visited = set()
    for u, v, t in sorted_edges:

        if u == v:
            continue

        src = str(u) + '-' + str(node_version[u])
        visited.add(u)
        new_nodes.add(src)

        if v not in visited:
            dst = str(v) + '-' + str(node_version[v])
            visited.add(v)
            new_nodes.add(dst)
            new_edges.append((src, dst, {'time' : int(t)}))
        else:
            dst_current = str(v) + '-' + str(node_version[v])
            dst_new = str(v) + '-' + str(node_version[v] + 1)
            node_version[v] += 1
            new_nodes.add(dst_new)
            new_edges.append((src, dst_new, {'time' : int(t)}))
            new_edges.append((dst_current, dst_new, {'time' : int(t)}))

    DAG = nx.DiGraph()
    DAG.add_nodes_from(list(new_nodes))
    DAG.add_edges_from(new_edges)

    return DAG

def test():
    # generate a test graph
    g = nx.MultiDiGraph()
    g.add_nodes_from([1,2,3,4])
    g.add_edges_from([
        (1, 2, {'time': 1}),
        (2, 3, {'time': 2}),
        (3, 1, {'time': 3}),
        (4, 1, {'time': 4}),
        (3, 3, {'time': 5}),
        (4, 1, {'time': 6}),
        (3, 4, {'time': 7}),
        (4, 3, {'time': 8}),
    ])

    # try converting
    dag = convert_DAG(g)
    for u, v, t in dag.edges(data=True):
        print(u, v, t)

    print(list(nx.all_simple_paths(dag, source='1-0', target='1-1')))

def main(cfg):
    sorted_tw_paths = sorted(os.listdir(os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, 'test')))
    tw_to_time = {}
    for tw, tw_file in enumerate(sorted_tw_paths):
        tw_to_time[tw] = tw_file[:-20]

    base_dir = cfg.preprocessing.build_graphs._graphs_dir

    tw = 0

    timestr = tw_to_time[tw]
    day = timestr[8:10].lstrip('0')
    graph_dir = os.path.join(base_dir, f"graph_{day}/{timestr}")

    test_graph = torch.load(graph_dir)

    dag = convert_DAG(test_graph)

    print(dag)
    print("converting finished")
    print(list(nx.simple_cycles(dag)))

if __name__ == "__main__":
    # args = get_runtime_required_args()
    # cfg = get_yml_cfg(args)
    #
    # main(cfg)

    test()
