import random
from datetime import timedelta

import networkx as nx
import torch

from provnet_utils import log, log_tqdm, get_all_files_from_folders


# The number of added edges is approx `NUM_MALICIOUS_PROCESSES * num_tws_containing_malicious_processes * NUM_UNAUTHORIZED_FILE_ACCESS`
NUM_ATTACKS = 1
NUM_MALICIOUS_PROCESSES = 5
NUM_UNAUTHORIZED_FILE_ACCESS = 20

def get_mean_time_delta(snapshot):
    # Extract and sort timestamps from edges
    timestamps = sorted(
        edge_data["time"] for _, _, _, edge_data in snapshot.edges(keys=True, data=True) if "time" in edge_data
    )
    if len(timestamps) < 2:
        return timedelta(seconds=0)  # No meaningful delta if fewer than 2 timestamps

    # Compute time deltas
    deltas = [
        (timestamps[i + 1] - timestamps[i]) / 1e9  # Convert nanoseconds to seconds
        for i in range(len(timestamps) - 1)
    ]
    mean_delta = sum(deltas) / len(deltas)  # Mean time delta in seconds
    return mean_delta, timestamps[0], timestamps[-1]

def get_days_from_cfg(cfg):
    """Extracts the list of days associated with train, val, and test splits."""
    return cfg.dataset["train_files"], cfg.dataset["val_files"], cfg.dataset["test_files"]

def load_graphs_for_days(base_dir, days):
    """Loads all graph snapshots for a given list of days."""
    return [torch.load(path) for day in days for path in get_all_files_from_folders(base_dir, [day])]

def get_processes_with_incoming_connections(graphs):
    """Finds processes with incoming 'EVENT_RECVFROM' edges in the given graphs."""
    processes = set()
    for graph in graphs:
        for u, v, key, data in graph.edges(keys=True, data=True):
            if data.get("label") == "EVENT_RECVFROM" and graph.nodes[v].get("node_type") == "subject":
                processes.add(v)
    return processes

def select_processes_with_constraints(train_graphs, val_graphs, test_graphs, num_processes):
    """
    Selects processes that:
    1. Have incoming network connections (EVENT_RECVFROM)
    2. Appear in all dataset splits (train, val, test)
    """
    # Get sets of processes with incoming network connections per dataset split
    train_processes = get_processes_with_incoming_connections(train_graphs)
    val_processes = get_processes_with_incoming_connections(val_graphs) # NOTE: may be empty on some datasets
    test_processes = get_processes_with_incoming_connections(test_graphs)

    # Step 1: Compute intersection to ensure selected processes exist in all splits
    valid_processes = train_processes & val_processes & test_processes

    if not valid_processes:
        raise RuntimeError("No processes found that exist in all dataset splits.")

    # Step 2: Randomly select up to NUM_MALICIOUS_PROCESSES
    selected_processes = random.sample(valid_processes, min(num_processes, len(valid_processes)))
    
    print(f"Selected processes: {selected_processes}")
    return selected_processes

def integrate_synthetic_attacks(graph_snapshots, cfg):
    """
    Integrates synthetic attack patterns into temporal provenance graphs.

    Parameters:
        graph_snapshots (list): List of temporal graph snapshots (MultiDiGraph objects).
    """
    # Combine all graph snapshots into a single graph for analysis
    combined_graph = nx.MultiDiGraph()
    for graph in graph_snapshots:
        combined_graph.add_edges_from(graph.edges(data=True))
        combined_graph.add_nodes_from(graph.nodes(data=True))

    # Load graph snapshots for each dataset split
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    train_graphs = load_graphs_for_days(base_dir, cfg.dataset.train_files)
    val_graphs = load_graphs_for_days(base_dir, cfg.dataset.val_files)
    test_graphs = load_graphs_for_days(base_dir, cfg.dataset.test_files)
    
    # TODO: select base on node degree or number of authrozed events
    selected_processes = select_processes_with_constraints(train_graphs, val_graphs, test_graphs, NUM_MALICIOUS_PROCESSES)
    
    selected_processes_paths = {n: combined_graph.nodes[n]["label"] for n in selected_processes}
    log(f"Selected processes for synthetic attacks:")
    for k, v in selected_processes_paths.items():
        log(f"{k}: {v}")
    log("")
    
    # Get all files in the dataset
    all_files = {
        node for node, data in combined_graph.nodes(data=True)
        if data.get("node_type") == "file"
    }
    
    tot_edges = 0
    tot_snapshots = []

    # Step 2: Integrate synthetic attacks
    for process in log_tqdm(selected_processes, desc="Synthetizing attacks"):
        
        # Get files the application interacts with
        interacted_files = {
            v for u, v, key, data in combined_graph.edges(process, keys=True, data=True)
            if combined_graph.nodes[v].get("node_type") == "file"
        }

        # Pick files outside of the intersection
        unauthorized_files = list(all_files - interacted_files)
        if not unauthorized_files:
            print(f"No unauthorized files for process {process}")
            continue

        i = -1
        for set_snapshots in [train_graphs, val_graphs, test_graphs]:
            for snapshot in set_snapshots:
                i += 1
                
                # All default edges are considered benign
                for u, v, key in snapshot.edges(keys=True):
                    snapshot.edges[u, v, key]["y"] = 0
            
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
                    unauthorized_files_in_snapshot, min(NUM_UNAUTHORIZED_FILE_ACCESS, len(unauthorized_files_in_snapshot))
                )
                log(f"Selected files for malicious access:")
                selected_files_paths = {n: combined_graph.nodes[n]["label"] for n in attack_files}
                for k, v in selected_files_paths.items():
                    log(f"{k}: {v}")
                
                mean_time_delta, min_time, max_time = get_mean_time_delta(snapshot)
                current_time = random.randint(min_time, max_time)

                # Add read/write interactions with unauthorized files
                for file in attack_files:
                    # Generate random times within the time window for each new edge
                    snapshot.add_edge(process, file, label="EVENT_OPEN", time=current_time, y=1)

                    current_time += mean_time_delta  # Add mean delta to simulate temporal progression
                    snapshot.add_edge(process, file, label="EVENT_READ", time=current_time, y=1)

                    current_time += mean_time_delta  # Add mean delta to simulate temporal progression
                    snapshot.add_edge(process, file, label="EVENT_WRITE", time=current_time, y=1)
                    tot_edges += 3

                # Add outgoing network activity if a suitable netflow exists
                outgoing_netflows = [
                    node for node, data in snapshot.nodes(data=True)
                    if data.get("node_type") == "netflow"
                ]
                if outgoing_netflows:
                    selected_netflow = random.choice(outgoing_netflows)
                    snapshot.add_edge(process, selected_netflow, label="EVENT_CONNECT", time=current_time+mean_time_delta, y=1)
                    tot_edges += 1
                    
                tot_snapshots.append(i)
                if len(tot_snapshots) % NUM_ATTACKS == 0:
                    break
                    

    log(f"Successfully added {tot_edges} synthetic edges in {len(tot_snapshots)}/{len(graph_snapshots)} snapshots.")
    log(f"Snapshots: {','.join(list(map(str, tot_snapshots)))}")
    
    return [*train_graphs, *val_graphs, *test_graphs]
