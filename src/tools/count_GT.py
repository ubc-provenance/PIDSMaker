import sys
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import wandb
from config import *
from provnet_utils import *
from preprocessing import (
    build_graphs,
)
from tqdm import tqdm
from labelling import get_ground_truth, get_uuid2nids
from detection.evaluation_methods.evaluation_utils import compute_tw_labels
from dataset_utils import get_rel2id

def compute_node_number(split_files, cfg):
    all_nids = set()
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    for graph_path in tqdm(sorted_paths, desc='Computing node number'):
        graph = torch.load(graph_path)
        all_nids |= set(graph.nodes())
    return all_nids

def get_events_between_GPs(cur,
               start_time,
               end_time,
               malicious_nodes : list):
    malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str}) AND dst_index_id IN ({malicious_nodes_str});"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_events_between_time_range(cur,
               start_time,
               end_time,):
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def generate_DAG(edges):
    node_version = {}
    for (u, v, t) in edges:
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
            new_edges.append((src, dst, {'time': int(t)}))
        else:
            dst_current = str(v) + '-' + str(node_version[v])
            dst_new = str(v) + '-' + str(node_version[v] + 1)
            node_version[v] += 1
            new_nodes.add(dst_new)
            new_edges.append((src, dst_new, {'time': int(t)}))
            new_edges.append((dst_current, dst_new, {'time': int(t)}))

    DAG = nx.DiGraph()
    DAG.add_nodes_from(list(new_nodes))
    DAG.add_edges_from(new_edges)

    return DAG, node_version

def dataset_counting(cfg):
    print("Start counting dataset nodes")

    node_in_train = compute_node_number(split_files=cfg.dataset.train_files, cfg=cfg)
    node_in_val = compute_node_number(split_files=cfg.dataset.val_files, cfg=cfg)
    node_in_test = compute_node_number(split_files=cfg.dataset.test_files, cfg=cfg)
    node_in_unused = compute_node_number(split_files=cfg.dataset.unused_files, cfg=cfg)

    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = set(str(nid) for nid in GP_nids)

    GP_num_in_files = len(GPs)
    log(f"There are {len(GPs)} GPs")

    GP_num_in_test_set = len(GPs & node_in_test)

    return node_in_train, node_in_val, node_in_test, node_in_unused, GP_num_in_files, GP_num_in_test_set

def select_events(cur,
                  start_time: str,
                  end_time: str,
                  ):

    sql = f"SELECT * FROM event_table WHERE timestamp_rec > {start_time} AND timestamp_rec < {end_time} ORDER BY timestamp_rec ;"

    cur.execute(sql)
    records = cur.fetchall()

    event2msg = {}

    for i in tqdm(records,desc='Selecting events'):
        src_index_id = str(i[1])
        operation = i[2]
        dst_index_id = str(i[4])
        event_uuid = i[5]
        timestamp = i[6]
        event_id = i[7]

        event2msg[event_uuid] = [src_index_id, operation, dst_index_id, timestamp, event_id]

    return event2msg

def get_n_hop_neighbors(graph, node, n):

    neighbors = set()
    current_level = {node}

    for _ in range(n):
        next_level = set()
        for current_node in current_level:
            next_level.update(graph.successors(current_node))
            next_level.update(graph.predecessors(current_node))
        neighbors.update(next_level)
        current_level = next_level

    neighbors.add(node)

    return neighbors

def gen_graph(start_time, end_time, cur):
    event2msg = select_events(cur, start_time, end_time)

    print(f"generate graph")
    edge_list = []
    for k, v in event2msg.items():
        src = v[0]
        dst = v[2]
        if src != dst:
            edge_list.append((src, dst))

    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)

    return graph

def get_n_hop_of_GP(graph, GPs, n):
    n_hop_of_GP = set()
    for gp in GPs:
        n_hop_of_GP.add(gp)

    for nid in graph.nodes():
        if nid in GPs:
            neighbors = get_n_hop_neighbors(graph, nid, n)
            n_hop_of_GP |= neighbors

    return n_hop_of_GP

def source_based_GT(cfg, cur, uuid2nids, rel2id):
    print("Start counting source based ground truth")

    attack_GPs = {}
    attack_to_des = {}
    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        print("==" * 30)
        print(f"start processing attak {attack}")

        print("get GPs")
        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))

        attack_GPs[attack] = set(ground_truth_nids)

        print("Get events between GPs")
        rows = get_events_between_GPs(cur, start_time, end_time, ground_truth_nids)

        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_between_GPs, _ = generate_DAG(edges)

        print("Get root nodes")
        root_nodes = set([node for node, in_degree in dag_between_GPs.in_degree() if in_degree == 0])

        print(f"Root nodes are: {root_nodes}")

        print("Get events in attack time range")
        rows = get_events_between_time_range(cur, start_time, end_time)
        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_of_attack, node_version = generate_DAG(edges)

        all_descendants = set()
        for root in root_nodes:
            descendants = nx.descendants(dag_of_attack, root)
            desc = set([v.split('-')[0] for v in descendants])
            all_descendants |= desc

        print(f"{len(all_descendants)} descedants of root nodes in the attack")

        attack_to_des[attack] = all_descendants

        print("==" * 30)

    root_desc = set()
    print("The number of descendants of root nodes:")
    for attack, desc_set in attack_to_des.items():
        print(f"{attack} : {len(desc_set)}")
        root_desc |= desc_set

    print(f"Total : {len(root_desc)}")

    return len(root_desc)

def neighbor_based_GT(cfg, cur):
    print("Start counting neighbor based ground truth")

    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [str(nid) for nid in GP_nids]

    neighbors = set()
    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        graph = gen_graph(start_time, end_time, cur)

        n =2

        neigh = get_n_hop_of_GP(graph, GPs, n)

        neighbors |= neigh

    return len(neighbors)

def batch_based_GT(cfg):
    print("Start counting batch based ground truth")
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)

    tw_to_malicious_nodes = compute_tw_labels(cfg)

    unique_nodes_in_attack_tw = set()
    node_number_in_attak_tw = 0
    for tw, nid2count in tw_to_malicious_nodes.items():
        if len(nid2count.items()) > 0:
            graph = torch.load(sorted_paths[tw])
            unique_nodes_in_attack_tw |= set(graph.nodes())
            node_number_in_attak_tw += len(graph.nodes())

    return len(unique_nodes_in_attack_tw)

def main(cfg, **kwargs):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}

    log("\n" + ("*" * 100))
    log("Tasks modified since last runs:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    log("\nTasks requiring re-execution:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    log(("*" * 100) + "\n")

    if should_restart["build_graphs"]:
        build_graphs.main(cfg)

    node_in_train, node_in_val, node_in_test, node_in_unused, GP_num_in_files, GP_num_in_test_set = dataset_counting(cfg)

    results = {
        'train node number': len(node_in_train),
        'val node number': len(node_in_val),
        'test node number': len(node_in_test),
        'unused node number': len(node_in_unused),
        'file GP number': GP_num_in_files,
        'test GP number': GP_num_in_test_set,
    }

    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    rel2id = get_rel2id(cfg)

    num_source_GT = source_based_GT(cfg, cur, uuid2nids, rel2id)
    results['source GP number'] = num_source_GT

    num_neighbors_GT = neighbor_based_GT(cfg, cur)
    results['neighbor GP number'] = num_neighbors_GT

    num_batch_GT = batch_based_GT(cfg)
    results['batch GP number'] = num_batch_GT

    for k, v in results.items():
        log(f"{k}: {v}")

    wandb.log(results)

if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    exp_name = args.exp if args.exp != "" else \
        "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="orthrus_main",
        name=exp_name,
        tags=tags,
    )

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)

    wandb.finish()