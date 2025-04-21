import multiprocessing
import os
import random

import numpy as np
from tqdm import tqdm


class TRW:
    def __init__(
        self,
        graph,
        walk_length,
        num_walks,
        workers,
        path_save_dir=None,
        time_weight="uniform",
        half_life=1,
    ):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.path_save_dir = path_save_dir
        self.time_weight = time_weight
        self.half_life = half_life

        self.forward_adj = self._gen_forward_adj_dict()
        self.backward_adj = self._get_backward_adj_dict()

        self.walk_paths = []

    def _gen_forward_adj_dict(self):
        forward_adj = {}
        for src, dst, k, attrs in tqdm(
            self.graph.edges(data=True, keys=True), desc="generating forward_adj dictionary"
        ):
            if src not in forward_adj:
                forward_adj[src] = {}
            if dst not in forward_adj[src]:
                forward_adj[src][dst] = []
            forward_adj[src][dst].append(attrs["time"])
        return forward_adj

    def _get_backward_adj_dict(self):
        backward_adj = {}
        for src, dst, k, attrs in tqdm(
            self.graph.edges(data=True, keys=True), desc="generating backward_adj dictionary"
        ):
            if dst not in backward_adj:
                backward_adj[dst] = {}
            if src not in backward_adj[dst]:
                backward_adj[dst][src] = []
            backward_adj[dst][src].append(attrs["time"])
        return backward_adj

    def run(self):
        workload_list = split_list(list(self.graph.nodes()), self.workers)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        arg_list = []
        for i in range(self.workers):
            args = (
                workload_list[i],
                self.walk_length,
                self.num_walks,
                self.forward_adj,
                self.backward_adj,
                lock,
                self.time_weight,
                self.half_life,
                self.path_save_dir,
            )
            arg_list.append(args)

        with multiprocessing.Pool(processes=self.workers) as pool:
            results = pool.starmap(parallel_generate_random_walk, arg_list)

        for result in results:
            self.walk_paths.extend(result)


def split_list(lst, n):
    avg = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result


def parallel_generate_random_walk(
    node_list,
    walk_length,
    num_walks,
    forward_adj,
    backward_adj,
    lock,
    time_weight,
    half_life=1,
    save_dir=None,
):
    walk_paths = []

    for node in tqdm(node_list, desc=f"{os.getpid()}: Generating random walks"):
        for i in range(num_walks):
            walk = []
            if node in forward_adj:
                # Start walking
                walk.append(node)
                last_time = -np.inf

                while len(walk) < walk_length:
                    if walk[-1] not in forward_adj:
                        break

                    walk_options = []
                    for neighbor, times in forward_adj[walk[-1]].items():
                        walk_options += [
                            (neighbor, edge_time) for edge_time in times if edge_time > last_time
                        ]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = min(map(lambda x: x[1], walk_options))

                    if time_weight == "uniform":
                        walk_to = random.choice(walk_options)
                    elif time_weight == "exponential":
                        time_probabilities = np.array(
                            list(
                                map(lambda x: np.exp((last_time - x[1]) / half_life), walk_options)
                            )
                        )
                        time_probabilities[np.isnan(time_probabilities)] = 0.0
                        time_probabilities /= sum(time_probabilities)
                        walk_to_idx = np.random.choice(
                            range(len(walk_options)), size=1, p=time_probabilities
                        )[0]
                        walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[1]
                    walk.append(walk_to[0])
            else:
                # Start walking
                walk.append(node)
                last_time = np.inf

                while len(walk) < walk_length:
                    if walk[-1] not in backward_adj:
                        break

                    walk_options = []
                    for neighbor, times in backward_adj[walk[-1]].items():
                        walk_options += [
                            (neighbor, edge_time) for edge_time in times if edge_time < last_time
                        ]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = max(map(lambda x: x[1], walk_options))

                    if time_weight == "uniform":
                        walk_to = random.choice(walk_options)
                    elif time_weight == "exponential":
                        time_probabilities = np.array(
                            list(
                                map(lambda x: np.exp((x[1] - last_time) / half_life), walk_options)
                            )
                        )
                        time_probabilities[np.isnan(time_probabilities)] = 0.0
                        time_probabilities /= sum(time_probabilities)
                        walk_to_idx = np.random.choice(
                            range(len(walk_options)), size=1, p=time_probabilities
                        )[0]
                        walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[1]
                    walk.insert(0, walk_to[0])
            walk_paths.append(walk)

    print(f"length: {len(walk_paths)}")

    if save_dir is not None:
        with lock:
            with open(save_dir, "a") as f:
                for path in walk_paths:
                    f.write(",".join(path) + "\n")

        print(f"Pid {os.getpid()}: walk paths saved to {save_dir}")

    return walk_paths
