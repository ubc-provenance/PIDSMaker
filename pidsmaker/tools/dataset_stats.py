import argparse
import csv
import os

import torch
import wandb
from tqdm import tqdm

import pidsmaker.labelling as labelling
from pidsmaker.config import get_yml_cfg
from pidsmaker.preprocessing import (
    build_graphs,
    transformation,
)
from pidsmaker.utils import (
    datetime_to_ns_time_US,
    get_all_files_from_folders,
    get_runtime_required_args,
    init_database_connection,
    log,
    remove_underscore_keys,
)


def stats_of_split(paths, split):
    nid_set = set()
    node_number = 0
    edge_number_list = []
    node_number_list = []
    in_degree_number = 0
    out_degree_number = 0

    for graph_path in tqdm(paths, desc=f"Processing split {split} set..."):
        graph = torch.load(graph_path)
        nid_set |= set(graph.nodes())
        node_number += len(graph.nodes())
        edge_number_list.append(graph.number_of_edges())
        node_number_list.append(graph.number_of_nodes())
        for node, in_deg in graph.in_degree():
            in_degree_number += in_deg
        for node, out_deg in graph.out_degree():
            out_degree_number += out_deg

    return (
        nid_set,
        node_number,
        edge_number_list,
        node_number_list,
        in_degree_number,
        out_degree_number,
    )


def get_stats(cfg):
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)
    val_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.val_files)
    test_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)
    unused_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.unused_files)

    (
        train_nodes,
        train_overlap_n_num,
        train_e_num_list,
        train_n_num_list,
        train_in_deg,
        train_out_deg,
    ) = stats_of_split(train_set_paths, "train")
    val_nodes, val_overlap_n_num, val_e_num_list, val_n_num_list, val_in_deg, val_out_deg = (
        stats_of_split(val_set_paths, "val")
    )
    test_nodes, test_overlap_n_num, test_e_num_list, test_n_num_list, test_in_deg, test_out_deg = (
        stats_of_split(test_set_paths, "test")
    )
    (
        unused_nodes,
        unused_overlap_n_num,
        unused_e_num_list,
        unused_n_num_list,
        unused_in_deg,
        unused_out_deg,
    ) = stats_of_split(unused_set_paths, "unused")

    results = {}

    all_nodes = train_nodes | val_nodes | test_nodes | unused_nodes
    results["total_node_number"] = len(all_nodes)
    results["train_node_number"] = len(train_nodes)
    results["val_node_number"] = len(val_nodes)
    results["test_node_number"] = len(test_nodes)
    results["unused_node_number"] = len(unused_nodes)

    total_in_deg = train_in_deg + val_in_deg + test_in_deg + unused_in_deg
    total_out_deg = train_out_deg + val_out_deg + test_out_deg + unused_out_deg
    results["avg_in_deg"] = total_in_deg / len(all_nodes)
    results["avg_out_deg"] = total_out_deg / len(all_nodes)

    edge_num_list = train_e_num_list + val_e_num_list + test_e_num_list + unused_e_num_list
    results["total_edge_number"] = sum(edge_num_list)
    results["train_edge_number"] = sum(train_e_num_list)
    results["val_edge_number"] = sum(val_e_num_list)
    results["test_edge_number"] = sum(test_e_num_list)
    results["unused_edge_number"] = sum(unused_e_num_list)

    results["avg_tw_edge_number"] = sum(edge_num_list) / len(edge_num_list)

    node_num_list = train_n_num_list + val_n_num_list + test_n_num_list + unused_n_num_list
    results["avg_tw_node_number"] = sum(node_num_list) / len(node_num_list)
    results["max_edges_per_tw"] = max(
        [*train_e_num_list, *val_e_num_list, *test_e_num_list, *unused_e_num_list]
    )

    return results


def count_mal_edges(cfg):
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = labelling.get_uuid2nids(cur)

    attack_index = 0
    results = {}

    for attack_tuple in cfg.dataset.attack_to_time_window:
        mal_edge_number = 0

        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))

        rows = labelling.get_events(cur, start_time, end_time)
        for row in rows:
            src_id = row[1]
            dst_id = row[4]

            if src_id in ground_truth_nids or dst_id in ground_truth_nids:
                mal_edge_number += 1

        results[attack_index] = mal_edge_number
        attack_index += 1

    return results


def malicious_stats(cfg):
    results = {}

    malicious_set, _, _ = labelling.get_ground_truth(cfg)
    results["total_mal_nodes"] = len(malicious_set)

    attack_to_nids = labelling.get_GP_of_each_attack(cfg)
    results["num_of_att"] = len(attack_to_nids.keys())
    for i in attack_to_nids.keys():
        results[f"att_{str(i)}_mal_nodes"] = len(attack_to_nids[i]["nids"])

    attack_to_edge_number = count_mal_edges(cfg)
    total_mal_edges = 0
    for k, v in attack_to_edge_number.items():
        results[f"att_{str(k)}_mal_edges"] = v
        total_mal_edges += v
    results["total_mal_edges"] = total_mal_edges

    return results


def main(cfg):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {
        subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps
    }

    log("\n" + ("*" * 100))
    log("Tasks modified since last runs:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    log("\nTasks requiring re-execution:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    log(("*" * 100) + "\n")

    if should_restart["build_graphs"]:
        build_graphs.main(cfg)

    if should_restart["transformation"]:
        transformation.main(cfg)

    results = {}
    stats = get_stats(cfg)
    for key, value in stats.items():
        results[key] = value

    results["train_set_days"] = cfg.dataset.train_files
    results["val_set_days"] = cfg.dataset.val_files
    results["test_set_days"] = cfg.dataset.test_files
    results["unused_set_days"] = cfg.dataset.unused_files

    mal_stats = malicious_stats(cfg)
    for key, value in mal_stats.items():
        results[key] = value

    print(f"Dataset stats of {cfg.dataset.name}")
    print("==" * 30)

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    exp_name = (
        args.exp
        if args.exp != ""
        else "|".join(
            [
                f"{k.split('.')[-1]}={v}"
                for k, v in args.__dict__.items()
                if "." in k and v is not None
            ]
        )
    )
    tags = args.tags.split(",") if args.tags != "" else [args.model]

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="dataset_stats",
        name=exp_name,
        tags=tags,
    )

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)

    wandb.finish()
