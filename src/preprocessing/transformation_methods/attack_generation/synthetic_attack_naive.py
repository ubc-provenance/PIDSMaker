import networkx as nx
import random
from datetime import timedelta

from provnet_utils import log, log_tqdm


# The number of added edges is approx `NUM_MALICIOUS_PROCESSES * num_tws_containing_malicious_processes * NUM_UNAUTHORIZED_FILE_ACCESS`
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

def integrate_synthetic_attacks(graph_snapshots):
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

    # Step 1: Pick an application with incoming network connections
    applications_with_incoming = [
        node for node, data in combined_graph.nodes(data=True)
        if data.get("node_type") == "subject" and any(
            combined_graph[u][v][key].get("label") == "EVENT_RECVFROM"
            for u, v, key in combined_graph.in_edges(node, keys=True)
        )
    ]

    if not applications_with_incoming:
        print("No applications with incoming network connections found.")
        return

    # TODO: select base on node degree or number of authrozed events
    selected_processes = random.sample(applications_with_incoming, min(NUM_MALICIOUS_PROCESSES, len(applications_with_incoming)))
    
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

        for i, snapshot in enumerate(graph_snapshots):
            # All default edges are considered benign
            for u, v, key in snapshot.edges(keys=True):
                snapshot.edges[u, v, key]["y"] = 0
        
            if process not in snapshot.nodes:
                continue
            tot_snapshots.append(i)

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

    log(f"Successfully added {tot_edges} synthetic edges in {len(tot_snapshots)}/{len(graph_snapshots)} snapshots.")
    log(f"Snapshots: {','.join(list(map(str, tot_snapshots)))}")
    return graph_snapshots
