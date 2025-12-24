from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.preprocessing.build_graph_methods import (
    build_default_graphs,
    build_magic_graphs,
)
from pidsmaker.utils.utils import get_multi_datasets


def main_from_config(cfg):
    graph_method = cfg.construction.used_method
    if graph_method == "default":
        build_default_graphs.main(cfg)
    elif graph_method == "magic":
        build_default_graphs.main(cfg)
        build_magic_graphs.main(cfg)
    else:
        raise ValueError(f"Unrecognized graph method: {graph_method}")


def main(cfg):
    multi_datasets = get_multi_datasets(cfg)
    if "none" in multi_datasets:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)

            if should_restart["construction"]:
                main_from_config(updated_cfg)
