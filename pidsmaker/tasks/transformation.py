import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from pidsmaker.config import SYNTHETIC_ATTACKS, update_cfg_for_multi_dataset
from pidsmaker.preprocessing.transformation_methods import (
    synthetic_attack_naive,
    transformation_dag,
    transformation_merge_edges,
    transformation_rcaid_pseudo_graph,
    transformation_undirected,
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
    graph = None
    if methods[0] == "merge_edges":
        # Merges edges from the stored columns, without building the graph of all the events
        cols = load_graph_columns(path)
        if cols is not None:
            graph = transformation_merge_edges.merge_columns(cols, cfg)
            del cols
            methods = methods[1:]
    if graph is None:
        graph = torch.load(path)

    # Apply all transformations to a single graph
    graph = apply_graph_transformations(graph, methods, cfg)

    file_name = path.split("/")[-1]
    dst_path = os.path.join(dst_dir, f"graph_{date}")
    os.makedirs(dst_path, exist_ok=True)
    save_graph(graph, os.path.join(dst_path, file_name))


# Transformations whose result does not depend on the random state (run in parallel)
_DETERMINISTIC_TRANSFORMATIONS = {"none", "merge_edges", "undirected", "dag"}

# State shared with the forked workers of `add_graph_transformation` (inherited, not pickled)
_WORKER_STATE = None


def _transform_file_worker(path, date):
    _transform_file(path, date, **_WORKER_STATE)


def add_graph_transformation(base_dir, dst_dir, cfg, methods):
    global _WORKER_STATE
    os.makedirs(dst_dir, exist_ok=True)

    dates = cfg.dataset.train_dates + cfg.dataset.val_dates + cfg.dataset.test_dates
    jobs = [(path, date) for date in dates for path in get_all_graphs_for_dates(base_dir, [date])]
    num_workers = min(get_num_workers(cfg), len(jobs))
    # Graphs are transformed independently, except by random transformations, which draw from
    # the global random state in graph order. A date in several splits is transformed once per
    # split as before, so it is done sequentially to not write the same files concurrently
    parallel = (
        num_workers > 1
        and _can_fork()
        and len(set(dates)) == len(dates)
        and set(methods) <= _DETERMINISTIC_TRANSFORMATIONS
    )
    if not parallel:
        for path, date in log_tqdm(jobs, desc="Transforming"):
            _transform_file(path, date, dst_dir, cfg, methods)
        return

    log(f"Transforming {len(jobs)} graphs ({num_workers} processes)...")
    _WORKER_STATE = dict(dst_dir=dst_dir, cfg=cfg, methods=methods)
    try:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_transform_file_worker, path, date) for path, date in jobs]
            for future in log_tqdm(as_completed(futures), desc="Transforming", total=len(jobs)):
                future.result()
    finally:
        _WORKER_STATE = None


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
