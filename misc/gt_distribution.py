import os

import matplotlib.pyplot as plt
import wandb

from pidsmaker.config import *
from pidsmaker.preprocessing import (
    build_graphs,
    transformation,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack, get_ground_truth
from pidsmaker.utils.utils import *


def node_distribution(cfg):
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    test_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    train_node_set = set()
    for train_path in train_set_paths:
        train_graph = torch.load(train_path)
        train_node_set |= set(train_graph.nodes())

    num_test = []
    num_unseen = []
    test_node_set = set()
    for test_set_path in test_set_paths:
        test_graph = torch.load(test_set_path)
        num_test.append(len(test_graph.nodes()))
        unseen_set = set(test_graph.nodes()) - train_node_set
        num_unseen.append(len(unseen_set))

        test_node_set |= set(test_graph.nodes())

    total_unseen = test_node_set - train_node_set
    print(f"There are {len(test_node_set)} unique nodes in test set")
    print(f"There are {len(total_unseen)} unseen nodes in test set")
    print(f"Unseen percentage is {len(total_unseen) / len(test_node_set) * 100:.2f}%")

    return num_test, num_unseen


def malicious_distribution(cfg, gt_nid_set, split="train"):
    graph_dir = cfg.preprocessing.transformation._graphs_dir
    test_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)
    train_set_paths = get_all_files_from_folders(graph_dir, cfg.dataset.train_files)

    print(f"There are {len(train_set_paths)} train set tws")
    print(f"There are {len(test_set_paths)} test set tws")

    if split == "train":
        checking_path_list = train_set_paths
    elif split == "test":
        checking_path_list = test_set_paths

    tw_to_num_mal = {}
    num_mal_list = []
    mal_to_tw = {}
    for tw, graph_path in enumerate(checking_path_list):
        num_mal = 0
        tw_graph = torch.load(graph_path)
        for nid in gt_nid_set:
            if nid not in mal_to_tw:
                mal_to_tw[nid] = []

            if str(nid) in tw_graph.nodes():
                num_mal += 1
                mal_to_tw[nid].append(tw)
        tw_to_num_mal[tw] = num_mal
        num_mal_list.append(num_mal)

    return num_mal_list, mal_to_tw


def plot_scatter(data, xlabel="", ylabel="", save_dir=None, title=""):
    x = list(range(len(data)))
    y = data

    plt.figure()
    plt.scatter(x, y, color="blue", alpha=0.5, marker=".")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_dir:
        plt.savefig(save_dir)
        print(f"Figure saved to {save_dir}")

    plt.show()


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

    out_dir = os.path.join("./results/ground_truth_distribution/", cfg.dataset.name)
    os.makedirs(out_dir, exist_ok=True)
    figure_out_dir = os.path.join(out_dir, "figures/")
    os.makedirs(figure_out_dir, exist_ok=True)

    gt_nid_set, gt_nid2paths, gt_uuid2nid = get_ground_truth(cfg)
    total_mal = len(gt_nid_set)
    print(f"There are {total_mal} malicious nodes")

    # tw_to_malicious_nodes = compute_tw_labels(cfg)

    train_malnum_list, mal_to_tw = malicious_distribution(cfg, gt_nid_set, split="train")
    train_save_dir = figure_out_dir + "num_mal_in_train_set_tws.png"
    plot_scatter(
        train_malnum_list,
        ylabel=f"Num of malicious nodes (/{total_mal})",
        xlabel="TWs",
        save_dir=train_save_dir,
        title="Malicious node number in train set",
    )

    mal_to_num_tw = {}
    for mn, tw_list in mal_to_tw.items():
        mal_to_num_tw[mn] = len(set(tw_list))

    sorted_items = sorted(mal_to_num_tw.items(), key=lambda x: x[1], reverse=True)
    attack_to_nids = get_GP_of_each_attack(cfg)
    for k, v in attack_to_nids.items():
        print(k, v)
    for i in sorted_items:
        malicious_nid = i[0]
        appear_in_attack = []
        for att, nid_set in attack_to_nids.items():
            if int(malicious_nid) in nid_set["nids"]:
                appear_in_attack.append(att)
        print(
            f"Node {i[0]} ({gt_nid2paths[i[0]]}) presents in {i[1]} training TWs, which appears in attack {appear_in_attack}"
        )

    num_test, num_unseen = node_distribution(cfg)
    unseen_percentage = [
        (unseen / test_num * 100) for unseen, test_num in zip(num_unseen, num_test)
    ]

    unseen_percentage_fig_dir = figure_out_dir + "unseen_percentage.png"
    plot_scatter(
        unseen_percentage,
        xlabel="TWs",
        ylabel="Percentage of unseen nodes (%)",
        save_dir=unseen_percentage_fig_dir,
    )


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
