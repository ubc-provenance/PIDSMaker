import multiprocessing

from tqdm import tqdm

import pidsmaker.parallel as parallel


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
            results = pool.starmap(parallel.parallel_generate_random_walk, arg_list)

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
