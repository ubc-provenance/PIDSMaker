import os
import random
import numpy as np
from tqdm import tqdm


def parallel_generate_random_walk(
                                  node_list,
                                  walk_length,
                                  num_walks,
                                  forward_adj,
                                  backward_adj,
                                  lock,
                                  time_weight,
                                  half_life = 1,
                                  save_dir = None
                                  ):
    walk_paths = []

    for node in tqdm(node_list,desc=f"{os.getpid()}: Generating random walks"):
        for i in range(num_walks):
            walk = []
            if node in forward_adj:
                #Start walking
                walk.append(node)
                last_time = -np.inf

                while len(walk) < walk_length:
                    if walk[-1] not in forward_adj:
                        break

                    walk_options = []
                    for neighbor, times in forward_adj[walk[-1]].items():
                        walk_options += [(neighbor, edge_time) for edge_time in times if edge_time > last_time]

                    #Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = min(map(lambda x: x[1], walk_options))

                    if time_weight == 'uniform':
                        walk_to = random.choice(walk_options)
                    elif time_weight == 'exponential':
                        time_probabilities = np.array(
                            list(map(lambda x: np.exp((last_time - x[1]) / half_life), walk_options)))
                        time_probabilities[np.isnan(time_probabilities)] = 0.0
                        time_probabilities /= sum(time_probabilities)
                        walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=time_probabilities)[0]
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
                        walk_options += [(neighbor, edge_time) for edge_time in times if edge_time < last_time]

                    # Skip dead end nodes
                    if len(walk_options) == 0:
                        break

                    if len(walk) == 1:
                        last_time = max(map(lambda x: x[1], walk_options))

                    if time_weight == 'uniform':
                        walk_to = random.choice(walk_options)
                    elif time_weight == 'exponential':
                        time_probabilities = np.array(
                            list(map(lambda x: np.exp((x[1] - last_time) / half_life), walk_options)))
                        time_probabilities[np.isnan(time_probabilities)] = 0.0
                        time_probabilities /= sum(time_probabilities)
                        walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=time_probabilities)[0]
                        walk_to = walk_options[walk_to_idx]

                    last_time = walk_to[1]
                    walk.insert(0, walk_to[0])
            walk_paths.append(walk)

    print(f"length: {len(walk_paths)}")

    if save_dir is not None:
        with lock:
            with open(save_dir, 'a') as f:
                for path in walk_paths:
                    f.write(','.join(path) + '\n')

        print(f"Pid {os.getpid()}: walk paths saved to {save_dir}")

    return walk_paths
