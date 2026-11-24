import contextlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from pidsmaker.config import SYNTHETIC_ATTACKS, update_cfg_for_multi_dataset
from pidsmaker.preprocessing.transformation_methods import (
    synthetic_attack_naive,
    transformation_dag,
    transformation_merge_edges,
    transformation_rcaid_pseudo_graph,
    transformation_undirected,
    transformation_utils,
)
from pidsmaker.preprocessing.build_graph_methods.build_default_graphs import _can_fork, get_num_workers
from pidsmaker.utils.graph_store import load_graph_columns, save_graph
from pidsmaker.utils.utils import (
    get_all_graphs_for_dates,
    get_multi_datasets,
    load_graphs_for_dates,
    log,
    log_start,
    log_tqdm,
    set_seed,
    symlink_directory,
)


def no_transformation(base_dir, dst_dir):
    # If no transformation is used, symlink to the original graphs directory
    symlink_directory(base_dir, dst_dir)


def add_synthetic_attacks(base_dir, dst_dir, cfg, method):
    base_dir = cfg.construction._graphs_dir
    train_graphs = load_graphs_for_dates(base_dir, cfg.dataset.train_dates)
    val_graphs = load_graphs_for_dates(base_dir, cfg.dataset.val_dates)

    processed_graphs = apply_synthetic_attacks(train_graphs, val_graphs, cfg, method)

    test_graphs = load_graphs_for_dates(base_dir, cfg.dataset.test_dates)
    graphs = [*processed_graphs, *test_graphs]

    os.makedirs(dst_dir, exist_ok=True)

    i = 0
    dates = cfg.dataset.train_dates + cfg.dataset.val_dates + cfg.dataset.test_dates
    for date in log_tqdm(dates, desc="Saving graphs to disk"):
        sorted_paths = get_all_graphs_for_dates(base_dir, [date])
        for path in sorted_paths:
            graph = graphs[i]
            i += 1

            file_name = path.split("/")[-1]
            dst_path = os.path.join(dst_dir, f"graph_{date}")
            os.makedirs(dst_path, exist_ok=True)
            save_graph(graph, os.path.join(dst_path, file_name))


def apply_synthetic_attacks(train_graphs, val_graphs, cfg, method):
    if method == "synthetic_attack_naive":
        return synthetic_attack_naive.main(train_graphs, val_graphs, cfg)

    raise ValueError(f"Invalid attack generation method {method}")


def _transform_file(path, date, dst_dir, cfg, methods):
    """Transforms the graph stored at `path` and saves it in `dst_dir`."""
    file_name = path.split("/")[-1]
    dst_path = os.path.join(dst_dir, f"graph_{date}")
    os.makedirs(dst_path, exist_ok=True)
    out_path = os.path.join(dst_path, file_name)

    graph = None
    if methods == ["rcaid_pseudo_graph"]:
        # Transforms the stored columns and saves the result without building networkx graphs
        cols = load_graph_columns(path)
        if cols is not None:
            transformation_rcaid_pseudo_graph.transform_columns_and_save(cols, cfg, out_path)
            return
    if methods[0] == "merge_edges":
        # Merges edges from the stored columns, without building the graph of all the events
        cols = load_graph_columns(path)
        if cols is not None:
            graph = transformation_merge_edges.merge_columns(cols, cfg)
            del cols
            methods = methods[1:]
    if graph is None:
        graph = torch.load(path)

    if methods and methods[-1] == "rcaid_pseudo_graph":
        # The last transformation saves its result without building it
        graph = apply_graph_transformations(graph, methods[:-1], cfg)
        transformation_rcaid_pseudo_graph.transform_and_save(graph, cfg, out_path)
        return

    # Apply all transformations to a single graph
    graph = apply_graph_transformations(graph, methods, cfg)
    save_graph(graph, out_path)


# Transformations whose result does not depend on the random state (run in parallel)
_DETERMINISTIC_TRANSFORMATIONS = {"none", "merge_edges", "undirected", "dag"}
# Transformations drawing from the global random state exactly once per graph (see
# `transformation_utils.randint_timestamps`). They run in parallel too, each graph drawing
# from the random state that a sequential run would have.
_ORDERED_RANDOM_TRANSFORMATIONS = {"rcaid_pseudo_graph"}

# State shared with the forked workers of `add_graph_transformation` (inherited, not pickled)
_WORKER_STATE = None


class _OtherTransformationFailed(Exception):
    pass


class _OrderedRandomState:
    """Lets graphs transformed in parallel draw from the global numpy random state in graph
    order, as in a sequential run: the draw of graph i starts from the state left by the draw of
    graph i - 1. Each graph must draw exactly once. Shared with forked workers."""

    def __init__(self, ctx):
        state = np.random.get_state()
        if state[0] != "MT19937":
            raise ValueError(f"Unsupported random state {state[0]}")
        self.cond = ctx.Condition()
        self.turn = ctx.Value("q", 0, lock=False)  # index of the graph allowed to draw
        self.failed = ctx.Value("b", 0, lock=False)
        self.key = ctx.Array("I", 624, lock=False)
        self.pos = ctx.Value("q", 0, lock=False)
        self.has_gauss = ctx.Value("q", 0, lock=False)
        self.gauss = ctx.Value("d", 0.0, lock=False)
        self._store(state)

    def _store(self, state):
        _, key, pos, has_gauss, gauss = state
        self.key[:] = key.tolist()
        self.pos.value, self.has_gauss.value, self.gauss.value = pos, has_gauss, gauss

    def _load(self):
        key = np.array(self.key[:], dtype=np.uint32)
        return ("MT19937", key, self.pos.value, self.has_gauss.value, self.gauss.value)

    @contextlib.contextmanager
    def draw(self, index):
        """Context in which graph `index` draws from the shared random state."""
        with self.cond:
            while self.turn.value != index:
                if self.failed.value:
                    raise _OtherTransformationFailed()
                self.cond.wait(timeout=1)
            np.random.set_state(self._load())
            try:
                yield
            except BaseException:
                self.fail()
                raise
            self._store(np.random.get_state())
            self.turn.value = index + 1
            self.cond.notify_all()

    def fail(self):
        with self.cond:
            self.failed.value = 1
            self.cond.notify_all()

    def final_state(self):
        return self._load()


def _transform_file_worker(index, path, date):
    rng = _WORKER_STATE["rng"]
    if rng is None:
        _transform_file(path, date, **_WORKER_STATE["args"])
        return
    drawn = []

    @contextlib.contextmanager
    def hook():
        if drawn:
            raise RuntimeError("A graph transformation drew random timestamps twice")
        drawn.append(True)
        with rng.draw(index):
            yield

    transformation_utils.random_state_hook = hook
    try:
        _transform_file(path, date, **_WORKER_STATE["args"])
    except BaseException:
        rng.fail()
        raise
    finally:
        transformation_utils.random_state_hook = None
    if not drawn:
        rng.fail()
        raise RuntimeError("A graph transformation did not draw random timestamps")


def add_graph_transformation(base_dir, dst_dir, cfg, methods):
    global _WORKER_STATE
    os.makedirs(dst_dir, exist_ok=True)

    dates = cfg.dataset.train_dates + cfg.dataset.val_dates + cfg.dataset.test_dates
    jobs = [(path, date) for date in dates for path in get_all_graphs_for_dates(base_dir, [date])]
    num_workers = min(get_num_workers(cfg), len(jobs))
    # Graphs are transformed independently. Random transformations draw from the global random
    # state in graph order, which is preserved if each graph draws once. A date in several
    # splits is transformed once per split as before, so it is done sequentially to not write
    # the same files concurrently
    num_random = sum(m in _ORDERED_RANDOM_TRANSFORMATIONS for m in methods)
    parallel = (
        num_workers > 1
        and _can_fork()
        and len(set(dates)) == len(dates)
        and set(methods) <= _DETERMINISTIC_TRANSFORMATIONS | _ORDERED_RANDOM_TRANSFORMATIONS
        and num_random <= 1
    )
    if not parallel:
        for path, date in log_tqdm(jobs, desc="Transforming"):
            _transform_file(path, date, dst_dir, cfg, methods)
        return

    log(f"Transforming {len(jobs)} graphs ({num_workers} processes)...")
    ctx = multiprocessing.get_context("fork")
    rng = _OrderedRandomState(ctx) if num_random else None
    _WORKER_STATE = dict(args=dict(dst_dir=dst_dir, cfg=cfg, methods=methods), rng=rng)
    try:
        # Graphs are started in order (a graph waiting for its turn to draw only waits for
        # graphs started before it)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_transform_file_worker, i, path, date)
                for i, (path, date) in enumerate(jobs)
            ]
            failed = False
            for future in log_tqdm(as_completed(futures), desc="Transforming", total=len(jobs)):
                if future.exception() is not None:
                    failed = True
                    if rng is not None:
                        rng.fail()
                    for f in futures:
                        f.cancel()
                    break
        if failed:
            # Error of the first failed graph (not of the graphs stopped because of it)
            errors = [f.exception() for f in futures if not f.cancelled() and f.exception()]
            errors.sort(key=lambda e: isinstance(e, _OtherTransformationFailed))
            raise errors[0]
    finally:
        _WORKER_STATE = None
    if rng is not None:
        # Random state at the end of a sequential run
        np.random.set_state(rng.final_state())


def apply_graph_transformations(graph, methods, cfg):
    for method in methods:
        if method == "none":
            pass
        elif method == "rcaid_pseudo_graph":
            graph = transformation_rcaid_pseudo_graph.main(graph, cfg)
        elif method == "undirected":
            graph = transformation_undirected.main(graph)
        elif method == "dag":
            graph = transformation_dag.main(graph)
        elif method == "merge_edges":
            graph = transformation_merge_edges.main(graph, cfg)
        else:
            raise ValueError(f"Unrecognized transformation method: {method}")

    return graph


def main_from_config(cfg):
    methods = cfg.transformation.used_methods
    methods = list(map(lambda x: x.strip(), methods.split(",")))

    base_dir = cfg.construction._graphs_dir
    dst_dir = cfg.transformation._graphs_dir

    if len(methods) == 1 and methods[0] == "none":
        no_transformation(base_dir, dst_dir)

    elif len(methods) == 1 and methods[0] in SYNTHETIC_ATTACKS.keys():
        add_synthetic_attacks(base_dir, dst_dir, cfg, methods[0])

    else:
        add_graph_transformation(base_dir, dst_dir, cfg, methods)


def main(cfg):
    set_seed(cfg)
    log_start(__file__)

    multi_datasets = get_multi_datasets(cfg)
    if "none" in multi_datasets:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)

            if should_restart["transformation"]:
                main_from_config(updated_cfg)
