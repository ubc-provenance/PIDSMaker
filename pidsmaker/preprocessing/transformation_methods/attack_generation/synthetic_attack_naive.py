import random
from collections import defaultdict

import networkx as nx

from pidsmaker.utils.utils import log, log_tqdm


def get_mean_time_delta(snapshot):
    """Copmputes the mean, min and max time delta in seconds between two edges in a graph."""
    timestamps = sorted(
        edge_data["time"]
        for _, _, _, edge_data in snapshot.edges(keys=True, data=True)
        if "time" in edge_data
    )
    deltas = [
        (timestamps[i + 1] - timestamps[i]) / 1e9  # Convert nanoseconds to seconds
        for i in range(len(timestamps) - 1)
    ]
    mean_delta = sum(deltas) / len(deltas)  # Mean time delta in seconds
    return mean_delta, timestamps[0], timestamps[-1]


def get_processes_with_incoming_connections(graphs, process_selection_method):
    """Finds processes with incoming 'EVENT_RECVFROM' edges in the given graphs."""
    processes = set()
    for graph in graphs:
        for u, v, key, data in graph.edges(keys=True, data=True):
            if process_selection_method == "random":
                if (
                    data.get("label") == "EVENT_RECVFROM"
                    and graph.nodes[v].get("node_type") == "subject"
                ):  # and graph.out_degree()[v] > 0: # we also select those we non-null out-degree or we
                    processes.add(v)

            else:
                raise ValueError(f"Invalid process selection method {process_selection_method}")
    return processes


def select_processes_with_constraints(
    train_graphs, val_graphs, num_processes, process_selection_method
):
    """Selects processes that have incoming network connections (EVENT_RECVFROM)"""
    train_processes = get_processes_with_incoming_connections(
        train_graphs, process_selection_method
    )
    val_processes = get_processes_with_incoming_connections(val_graphs, process_selection_method)

    valid_processes = train_processes & val_processes

    if not valid_processes:
        raise RuntimeError("No processes found that exist in both dataset splits.")

    selected_processes = random.sample(valid_processes, min(num_processes, len(valid_processes)))

    return selected_processes


def main(train_graphs, val_graphs, cfg):
    """Integrates synthetic attack patterns into temporal provenance graphs."""
    num_attacks = cfg.transformation.synthetic_attack_naive.num_attacks
    num_malicious_process = cfg.transformation.synthetic_attack_naive.num_malicious_process
    num_unauthorized_file_access = (
        cfg.transformation.synthetic_attack_naive.num_unauthorized_file_access
    )
    process_selection_method = cfg.transformation.synthetic_attack_naive.process_selection_method

    # Combine all graph snapshots into a single graph for analysis
    combined_graph = nx.MultiDiGraph()
    for graph in [*train_graphs, *val_graphs]:
        combined_graph.add_edges_from(graph.edges(data=True))
        combined_graph.add_nodes_from(graph.nodes(data=True))

        # All default edges are considered benign
        for u, v, key in graph.edges(keys=True):
            graph.edges[u, v, key]["y"] = 0

    # TODO: select base on node degree or number of authrozed events
    selected_processes = select_processes_with_constraints(
        train_graphs, val_graphs, num_malicious_process, process_selection_method
    )

    selected_processes_paths = {n: combined_graph.nodes[n]["label"] for n in selected_processes}
    log("Selected processes for synthetic attacks:")
    for k, v in selected_processes_paths.items():
        log(f"{k}: {v}")
    log("")

    # Get all files in the dataset
    all_files = {
        node for node, data in combined_graph.nodes(data=True) if data.get("node_type") == "file"
    }

    tot_edges = 0
    tot_snapshots = []

    # Integrate synthetic attacks
    for process in log_tqdm(selected_processes, desc="Synthetizing attacks"):
        # Get files the application interacts with
        interacted_files = {
            v
            for u, v, key, data in combined_graph.edges(process, keys=True, data=True)
            if combined_graph.nodes[v].get("node_type") == "file"
        }

        # Pick files outside of the intersection
        unauthorized_files = list(all_files - interacted_files)
        if not unauthorized_files:
            print(f"No unauthorized files for process {process}")
            continue

        split2paths = defaultdict(lambda: defaultdict(list))
        i = -1
        for snapshots, split in [(train_graphs, "train"), (val_graphs, "val")]:
            for snapshot in snapshots:
                i += 1

                if process not in snapshot.nodes:
                    continue

                # Ensure unauthorized files exist in this snapshot
                unauthorized_files_in_snapshot = [
                    file for file in unauthorized_files if file in snapshot.nodes
                ]
                if not unauthorized_files_in_snapshot:
                    print(f"No unauthorized files for process {process} in snapshot.")
                    continue

                # Randomly select files for the attack
                attack_files = random.sample(
                    unauthorized_files_in_snapshot,
                    min(num_unauthorized_file_access, len(unauthorized_files_in_snapshot)),
                )

                selected_files_paths = {n: combined_graph.nodes[n]["label"] for n in attack_files}
                for node_id, path in selected_files_paths.items():
                    split2paths[split][i].append(
                        (f"{process}->{node_id}\t {combined_graph.nodes[process]['label']}->{path}")
                    )

                mean_time_delta, min_time, max_time = get_mean_time_delta(snapshot)
                current_time = random.randint(min_time, max_time)

                # Add read/write interactions with unauthorized files
                for file in attack_files:
                    # Generate random times within the time window for each new edge
                    snapshot.add_edge(process, file, label="EVENT_OPEN", time=current_time, y=1)

                    current_time += (
                        mean_time_delta  # Add mean delta to simulate temporal progression
                    )
                    snapshot.add_edge(process, file, label="EVENT_READ", time=current_time, y=1)

                    current_time += (
                        mean_time_delta  # Add mean delta to simulate temporal progression
                    )
                    snapshot.add_edge(process, file, label="EVENT_WRITE", time=current_time, y=1)
                    tot_edges += 3

                # Add outgoing network activity if a suitable netflow exists
                outgoing_netflows = [
                    node
                    for node, data in snapshot.nodes(data=True)
                    if data.get("node_type") == "netflow"
                ]
                if outgoing_netflows:
                    selected_netflow = random.choice(outgoing_netflows)
                    snapshot.add_edge(
                        process,
                        selected_netflow,
                        label="EVENT_CONNECT",
                        time=current_time + mean_time_delta,
                        y=1,
                    )
                    tot_edges += 1

                tot_snapshots.append(i)
                if len(tot_snapshots) % num_attacks == 0:
                    break

        for split, items in split2paths.items():
            log(f"Split: {split}", pre_return_line=True)
            for snapshot, l in items.items():
                log(f"  Snapshot {snapshot}:")
                for event in l:
                    log(f"    {event}")

    log(
        f"Successfully added {tot_edges} synthetic edges in {len(tot_snapshots)}/{len(train_graphs) + len(val_graphs)} snapshots."
    )
    log(f"Snapshots: {','.join(list(map(str, tot_snapshots)))}")

    return [*train_graphs, *val_graphs]
