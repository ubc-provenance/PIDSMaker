import argparse
import hashlib
import os
import pathlib
import sys
import uuid
from collections import OrderedDict
from copy import deepcopy

import yaml
from yacs.config import CfgNode as CN

from .config import (
    DATASET_DEFAULT_CONFIG,
    DECODERS_EDGE_LEVEL,
    DECODERS_NODE_LEVEL,
    EXPERIMENTS_CONFIG,
    OBJECTIVES_EDGE_LEVEL,
    OBJECTIVES_NODE_LEVEL,
    SYNTHETIC_ATTACKS,
    TASK_ARGS,
    TASK_DEPENDENCIES,
    UNCERTAINTY_EXP_YML_FOLDER,
    Arg,
)

ROOT_PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()
ROOT_GROUND_TRUTH_DIR = os.path.join(ROOT_PROJECT_PATH, "Ground_Truth/")

# ================================================================================


def get_default_cfg(args):
    """
    Inits the shared cfg object with default configurations.
    """
    cfg = CN()
    cfg._artifact_dir = args.artifact_dir

    cfg._test_mode = args.test_mode
    cfg._debug = not args.wandb
    cfg._is_running_mc_dropout = False

    cfg._force_restart = args.force_restart
    cfg._use_cpu = args.cpu
    cfg._model = args.model
    cfg._tuning_mode = args.tuning_mode
    cfg._experiment = args.experiment
    cfg._tuning_file_path = args.tuning_file_path
    cfg._include_yml = None
    cfg._exp = args.exp

    cfg._restart_from_scratch = args.restart_from_scratch
    if cfg._restart_from_scratch:
        cfg._run_random_seed = str(uuid.uuid4())

    # Database: we simply create variables for all configurations described in the dict
    cfg.database = CN()
    cfg.database.host = args.database_host
    cfg.database.user = args.database_user
    cfg.database.password = args.database_password
    cfg.database.port = args.database_port

    # Dataset: we simply create variables for all configurations described in the dict
    set_dataset_cfg(cfg, args.dataset)

    # Tasks: we create nested None variables for all arguments
    def create_cfg_recursive(cfg, task_args_dict: dict):
        for task, subtasks in task_args_dict.items():
            if isinstance(subtasks, dict):
                setattr(cfg, task, CN())
                task_cfg = getattr(cfg, task)
                create_cfg_recursive(task_cfg, dict(subtasks.items()))
            else:
                setattr(cfg, task, None)

    create_cfg_recursive(cfg, TASK_ARGS)

    # Experiments
    create_cfg_recursive(cfg, EXPERIMENTS_CONFIG)

    return cfg


def set_dataset_cfg(cfg, dataset):
    cfg.dataset = CN()
    cfg.dataset.name = dataset
    for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
        setattr(cfg.dataset, attr, value)


def get_runtime_required_args(return_unknown_args=False, args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of the model")
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument(
        "--force_restart",
        type=str,
        default="",
        help="The subtask or subtasks from which to restart",
    )
    parser.add_argument(
        "--restart_from_scratch",
        action="store_true",
        help="Starts pipeline in a fresh new task path",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to submit logs to wandb")
    parser.add_argument(
        "--project", type=str, default="PIDSMaker", help="Name of the wandb project"
    )
    parser.add_argument("--exp", type=str, default="", help="Name of the experiment")
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Name of the tag to use. Tags are used to group runs together",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Whether to run the framework on CPU rather than GPU"
    )
    parser.add_argument(
        "--experiment", type=str, default="none", help="The experiment yml config file"
    )
    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="none",
        help="Name of the tuning mode to run the pipeline with wandb sweeps",
    )
    parser.add_argument(
        "--tuned", action="store_true", help="Whether to load the best fine-tuned hyperparameters"
    )
    parser.add_argument(
        "--tuning_file_path", default="", help="If set, use the given YML path for tuning"
    )
    parser.add_argument(
        "--database_host", default="postgres", help="Host machine where the db is located"
    )
    parser.add_argument(
        "--database_user", default="postgres", help="Database user to connect to the database"
    )
    parser.add_argument(
        "--database_password", default="postgres", help="The password to the database user"
    )
    parser.add_argument(
        "--database_port", default="5432", help="The port number for Postgres (default: 5432)"
    )
    parser.add_argument("--sweep_id", default="", help="ID of a wandb sweep for multi-agent runs")
    parser.add_argument(
        "--artifact_dir", default="/home/artifacts/", help="Destination folder for generated files"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Whether to run the framework as in functional tests.",
    )

    # Script-specific args
    parser.add_argument("--show_attack", type=int, help="Number of attack for plotting", default=0)
    parser.add_argument("--gt_type", type=str, help="Type of ground truth", default="orthrus")
    parser.add_argument("--plot_gt", type=bool, help="If we plot ground truth", default=False)

    # All args in the cfg can be also set in the arg parser from CLI
    all_args = {
        **TASK_ARGS,
        **EXPERIMENTS_CONFIG,
    }
    parser = add_cfg_args_to_parser(all_args, parser)

    try:
        args, unknown_args = parser.parse_known_args(args)
    except:
        parser.print_help()
        sys.exit(1)

    if return_unknown_args:
        return args, unknown_args
    return args


def overwrite_cfg_with_args(cfg, args):
    """
    The framework can be also parametrized using the CLI args.
    These args are priorited compared to yml file parameters.
    This function simply overwrites the cfg with the parameters
    given within args.

    To override a parameter in cfg, use a dotted style:
    ```python pidsmaker/main.py --gnn_training.seed=42```
    """
    for arg, value in args.__dict__.items():
        if "." in arg and value is not None:
            cfg_ptr = cfg
            dots = arg.split(".")
            path, attr_name = dots[:-1], dots[-1]

            for attr in path:
                cfg_ptr = getattr(cfg_ptr, attr)
            setattr(cfg_ptr, attr_name, value)


def set_shortcut_variables(cfg):
    cfg._is_node_level = cfg.gnn_training.decoder.used_methods is not None and any(
        [
            method
            for method in OBJECTIVES_NODE_LEVEL
            if method in cfg.gnn_training.decoder.used_methods
        ]
    )


def set_task_paths(cfg, subtask_concat_value=None):
    subtask_to_hash = {}
    # Directories common to all tasks
    for task in TASK_ARGS:
        task_cfg = getattr(cfg, task)
        restart_values = flatten_arg_values(task_cfg)
        if (
            task == "build_graphs"
        ):  # to restart from beginning if train files are changed
            restart_values += cfg.dataset.train_files
            if (
                cfg._restart_from_scratch
            ):  # to start from a brand new folder, we generate a random id to add to the hash
                restart_values += [cfg._run_random_seed]
        if subtask_concat_value is not None:
            if task == subtask_concat_value["subtask"]:
                restart_values += [subtask_concat_value["concat_value"]]

        clean_hash_args = [
            "".join([c for c in str(arg) if c not in set(" []\"'")])
            for arg in restart_values
            if not arg.startswith("_")
        ]
        final_hash_string = ",".join(clean_hash_args)
        final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()

        subtask_to_hash[task] = final_hash_string

    # Then, for each subtask, we want its unique hash to also depend from its previous dependencies' hashes.
    # For example, if I run the same subtask A two times, with two different subtasks B and C, the results
    # would be different and would be stored in the same folder A if we don't consider the hash of B and C.
    for task in TASK_ARGS:
        task_cfg = getattr(cfg, task)
        deps = sorted(list(get_dependees(task, TASK_DEPENDENCIES, set())))
        deps_hash = "".join([subtask_to_hash[dep] for dep in deps])

        final_hash_string = deps_hash + subtask_to_hash[task]
        final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()

        if task in ["build_graphs", "transformation", "feat_training", "feat_inference"]:
            task_cfg._task_path = os.path.join(
                cfg._artifact_dir, task, cfg.dataset.name, task, final_hash_string
            )
        else:
            task_cfg._task_path = os.path.join(
                cfg._artifact_dir, task, task, final_hash_string, cfg.dataset.name
            )

        # The directory to save logs related to the preprocessing task
        task_cfg._logs_dir = os.path.join(task_cfg._task_path, "logs/")
        os.makedirs(task_cfg._logs_dir, exist_ok=True)

    # Preprocessing paths
    cfg.build_graphs._graphs_dir = os.path.join(
        cfg.build_graphs._task_path, "nx/"
    )
    cfg.build_graphs._tw_labels = os.path.join(
        cfg.build_graphs._task_path, "tw_labels/"
    )
    cfg.build_graphs._node_id_to_path = os.path.join(
        cfg.build_graphs._task_path, "node_id_to_path/"
    )
    cfg.build_graphs._dicts_dir = os.path.join(
        cfg.build_graphs._task_path, "indexid2msg/"
    )
    cfg.build_graphs._mimicry_dir = os.path.join(
        cfg.build_graphs._task_path, "mimicry/"
    )
    cfg.build_graphs._magic_dir = os.path.join(
        cfg.build_graphs._task_path, "magic/"
    )
    cfg.build_graphs._magic_graphs_dir = os.path.join(
        cfg.build_graphs._magic_dir, "dgl_graphs/"
    )

    cfg.transformation._graphs_dir = os.path.join(
        cfg.transformation._task_path, "nx/"
    )

    # Featurization paths
    cfg.feat_training._model_dir = os.path.join(
        cfg.feat_training._task_path, "stored_models/"
    )
    cfg.feat_training.temporal_rw._random_walk_dir = os.path.join(
        cfg.feat_training._task_path, "random_walks/"
    )
    cfg.feat_training.temporal_rw._random_walk_corpus_dir = os.path.join(
        cfg.feat_training.temporal_rw._random_walk_dir, "random_walk_corpus/"
    )
    cfg.feat_training.alacarte._random_walk_dir = os.path.join(
        cfg.feat_training._task_path, "random_walks/"
    )
    cfg.feat_training.alacarte._random_walk_corpus_dir = os.path.join(
        cfg.feat_training.alacarte._random_walk_dir, "random_walk_corpus/"
    )
    cfg.feat_training.alacarte._vec_graphs_dir = os.path.join(
        cfg.feat_training._task_path, "vectorized/"
    )

    cfg.feat_inference._edge_embeds_dir = os.path.join(
        cfg.feat_inference._task_path, "edge_embeds/"
    )
    cfg.feat_inference._model_dir = os.path.join(
        cfg.feat_inference._task_path, "stored_models/"
    )

    # Detection paths
    cfg.graph_preprocessing._preprocessed_graphs_dir = os.path.join(
        cfg.graph_preprocessing._task_path, "preprocessed_graphs/"
    )

    cfg.gnn_training._trained_models_dir = os.path.join(
        cfg.gnn_training._task_path, "trained_models/"
    )
    cfg.gnn_training._edge_losses_dir = os.path.join(
        cfg.gnn_training._task_path, "edge_losses/"
    )
    cfg.gnn_training._magic_dir = os.path.join(
        cfg.gnn_training._task_path, "magic/"
    )
    cfg.evaluation._precision_recall_dir = os.path.join(
        cfg.evaluation._task_path, "precision_recall_dir/"
    )
    cfg.evaluation._uncertainty_exp_dir = os.path.join(
        cfg.evaluation._task_path, "uncertainty_exp/"
    )
    cfg.evaluation.queue_evaluation._precision_recall_dir = os.path.join(
        cfg.evaluation._task_path, "precision_recall_dir/"
    )
    cfg.evaluation.queue_evaluation._queues_dir = os.path.join(
        cfg.evaluation._task_path, "queues_dir/"
    )
    cfg.evaluation.queue_evaluation._predicted_queues_dir = os.path.join(
        cfg.evaluation._task_path, "predicted_queues_dir/"
    )
    cfg.evaluation.queue_evaluation._kairos_dir = os.path.join(
        cfg.evaluation._task_path, "kairos_dir/"
    )
    cfg.evaluation._results_dir = os.path.join(
        cfg.evaluation._task_path, "results/"
    )

    # Ground Truth paths
    cfg._ground_truth_dir = os.path.join(
        ROOT_GROUND_TRUTH_DIR, cfg.evaluation.ground_truth_version + "/"
    )

    # Triage paths
    cfg.triage._tracing_graph_dir = os.path.join(
        cfg.triage._task_path, "tracing_graphs"
    )


def validate_yml_file(yml_file: str, dictionary: dict):
    with open(yml_file, "r") as file:
        user_config = yaml.safe_load(file)

    def validate_config(user_config, tasks, path=None):
        if path is None:
            path = []
        if not user_config:
            raise ValueError(f"Config at {' > '.join(path)} is empty but should not be.")

        for key, sub_tasks in tasks.items():
            if key in user_config:
                sub_config = user_config[key]
                if isinstance(sub_tasks, dict):
                    validate_config(sub_config, sub_tasks, path + [key])
                else:
                    if sub_config is None:
                        raise ValueError(
                            f"Parameter '{' > '.join(path + [key])}' should not be None."
                        )
                    expected_type = sub_tasks.type

                    if not isinstance(sub_config, expected_type):
                        raise TypeError(
                            f"Parameter '{' > '.join(path + [key])}' should be of type {expected_type.__name__}."
                        )

                    expected_vals = sub_tasks.vals
                    if expected_vals is not None:
                        user_literal_str = list(map(lambda x: x.strip(), sub_config.split(",")))
                        for e in user_literal_str:
                            if e not in expected_vals:
                                raise ValueError(
                                    f"Invalid argument {key} with value {e}. Expected values: {expected_vals}"
                                )

    validate_config(user_config, dictionary)


def check_args(args):
    available_models = os.listdir(os.path.join(ROOT_PROJECT_PATH, "config"))
    if not any([args.model in model for model in available_models]):
        raise ValueError(f"Unknown model {args.model}. Available models are {available_models}")

    available_datasets = DATASET_DEFAULT_CONFIG.keys()
    if args.dataset not in available_datasets:
        raise ValueError(
            f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}"
        )


def get_yml_file(filename, folder=""):
    return os.path.join(ROOT_PROJECT_PATH, "config", folder, f"{filename}.yml")


def merge_cfg_and_check_syntax(cfg, yml_file, syntax_check=TASK_ARGS):
    user_config = CN(load_yml_file_recursive(yml_file, syntax_check=syntax_check))
    cfg.merge_from_other_cfg(user_config)
    return cfg


def load_yml_file_recursive(yml_file, syntax_check=TASK_ARGS):
    validate_yml_file(yml_file, syntax_check)
    with open(yml_file, "r") as file:
        user_config = yaml.safe_load(file)
    if "_include_yml" in user_config:
        yml_to_include = get_yml_file(user_config["_include_yml"])
        included_config = load_yml_file_recursive(yml_to_include, syntax_check) or {}
        user_config = deep_merge_dicts(included_config, user_config)
    return user_config


def deep_merge_dicts(target, source):
    for k, v in source.items():
        target[k] = (
            deep_merge_dicts(target[k], v)
            if isinstance(v, dict) and isinstance(target.get(k), dict)
            else v
        )
    return target


def get_yml_cfg(args):
    # Checks that CLI args are OK
    check_args(args)

    # Inits with default configurations
    cfg = get_default_cfg(args)
    
    # Checks that all configurations are valid and merge yml file to cfg
    yml_file = get_yml_file(args.model)
    merge_cfg_and_check_syntax(cfg, yml_file)

    # Overrides with best hyperparameters
    if args.tuned:
        tuning_file = (
            "orthrus" if args.model in ["orthrus_non_snooped", "orthrus_fixed"] else args.model
        )
        if args.model == "orthrus_fixed" and args.dataset == "CLEARSCOPE_E3":  # speciifc case
            tuning_file = args.model
        tuned_yml_file = get_yml_file(
            f"tuned_{tuning_file}", folder=f"tuned_baselines/{cfg.dataset.name.lower()}/"
        )
        merge_cfg_and_check_syntax(cfg, tuned_yml_file)

    # Same for experiments
    exp_yml_file = get_yml_file(os.path.join(UNCERTAINTY_EXP_YML_FOLDER, args.experiment))
    merge_cfg_and_check_syntax(cfg, exp_yml_file, syntax_check=EXPERIMENTS_CONFIG)

    # Overwrites args to the cfg
    overwrite_cfg_with_args(cfg, args)

    # Here we create some variables based on parameters for easier usage
    set_shortcut_variables(cfg)

    # Based on the defined restart args, computes a unique path on disk
    # to store the files of each task
    set_task_paths(cfg)

    # Calculates which subtasks have to be re-executed
    set_subtasks_to_restart(yml_file, cfg)

    # Yield errors if some combinations of parameters are not possible
    check_edge_cases(cfg)

    return cfg


def check_edge_cases(cfg):
    """
    We want to check all errors prior to running the framework here.
    Yield EnvironmentError to be handled in tests.
    """
    decoders = cfg.gnn_training.decoder.used_methods
    use_tgn_neigh_loader = (
        "tgn_last_neighbor" in cfg.graph_preprocessing.intra_graph_batching.used_methods
    )
    use_tgn = "tgn" in cfg.gnn_training.encoder.used_methods
    use_rcaid_pseudo_graph = "rcaid_pseudo_graph" in cfg.transformation.used_methods

    if use_tgn_neigh_loader:
        if use_rcaid_pseudo_graph:
            raise ValueError(
                "Cannot use TGN with RCaid pseudo graph transformation. Edge timestamps are ignored with this transformation."
            )
        if not use_tgn:
            raise ValueError("Couldn't use `tgn_last_neighbor` without `tgn` as encoder.")

    if use_tgn:
        if not use_tgn_neigh_loader:
            raise ValueError("Couldn't use `tgn` as encoder without `tgn_last_neighbor` as loader.")
        if cfg.graph_preprocessing.inter_graph_batching.used_method != "none":
            raise ValueError("TGN-based encoders do not support inter graph batching yet.")

    if use_rcaid_pseudo_graph:
        if "predict_edge_type" in decoders:
            raise ValueError(
                "Cannot predict edge type as it is removed in the pseudo graph transformation"
            )

    if cfg.feat_training.used_method == "fasttext":
        if cfg.feat_training.fasttext.use_pretrained_fb_model:
            emb_dim = cfg.feat_training.emb_dim
            if emb_dim != 300:
                raise ValueError(
                    f"Invalid `emb_dim={emb_dim}`, should be set to 300 if `use_pretrained_fb_model=True`."
                )

    if "reconstruct_masked_features" in decoders or "predict_masked_struct" in decoders:
        if cfg.evaluation.node_evaluation.threshold_method != "magic":
            raise ValueError("These decoders are only working with magic thresholding yet.")

    if cfg.gnn_training.decoder.use_few_shot:
        if cfg.transformation.used_methods not in SYNTHETIC_ATTACKS.keys():
            raise ValueError(
                "Few-shot mode requires an attack generation method within `preprocessing.transformation.used_methods`"
            )

    use_multi_dataset = "none" not in cfg.build_graphs.multi_dataset
    if cfg.feat_training.multi_dataset_training and use_multi_dataset:
        method = cfg.feat_training.used_method.strip()
        if method not in ["word2vec", "fasttext", "hierarchical_hashing", "only_type"]:
            raise NotImplementedError(f"Multi-dataset mode not implemented for method {method}")
    if (
        cfg.feat_training.multi_dataset_training
        or cfg.graph_preprocessing.multi_dataset_training
    ):
        if not use_multi_dataset:
            raise ValueError(
                "Using multi-dataset mode requires setting `preprocessing.build_graphs.multi_dataset`"
            )

    if cfg.evaluation.used_method == "edge_evaluation":
        if cfg._is_node_level:
            raise ValueError("Edge evaluation not implemented for node-level detection.")


def set_subtasks_to_restart(yml_file: str, cfg):
    """
    Given a cfg, returns a boolean for each subtask, being `True` if
    the subtask requires to be restarted and `False` if the current arguments
    do not require a restart.
    In practice, we restart a subtask if there is no TASK_FINISHED_FILE in its `_task_path`.
    """
    user_config = load_yml_file_recursive(yml_file)
    tasks_in_yml_file = set(
        [task for task in user_config if not task.startswith("_")]
    )

    should_restart = OrderedDict()
    for task in TASK_ARGS:
        if task in tasks_in_yml_file:
            task_cfg = getattr(cfg, task)
            existing_files = [files for _, _, files in os.walk(task_cfg._task_path)]
            has_finished = any(
                [
                    files
                    for files in existing_files
                    for f in files
                    if f.endswith(TASK_FINISHED_FILE)
                ]
            )
            should_restart[task] = not has_finished
        else:
            should_restart[task] = False

    should_restart_with_deps = get_subtasks_to_restart_with_dependencies(
        should_restart, TASK_DEPENDENCIES, cfg._force_restart
    )

    # Dicts are not accepted in the cfg
    should_restart = [(subtask, restart) for subtask, restart in should_restart.items()]
    should_restart_with_deps = [
        (subtask, restart) for subtask, restart in should_restart_with_deps.items()
    ]

    cfg._subtasks_should_restart = should_restart
    cfg._subtasks_should_restart_with_deps = should_restart_with_deps


def update_task_paths_to_restart(cfg, subtask_concat_value=None):
    """Simply recomputes if tasks should be restarted."""
    yml_file = get_yml_file(cfg._model)
    set_dataset_cfg(cfg, cfg.dataset.name)
    set_shortcut_variables(cfg)
    set_task_paths(cfg, subtask_concat_value=subtask_concat_value)
    set_subtasks_to_restart(yml_file, cfg)
    should_restart = {
        subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps
    }
    check_edge_cases(cfg)
    return should_restart


def update_cfg_for_multi_dataset(cfg, dataset):
    cfg = deepcopy(cfg)
    cfg.dataset.name = dataset
    #     cfg.build_graphs.multi_dataset = "none"
    should_restart = update_task_paths_to_restart(cfg)
    return cfg, should_restart


def get_dependencies(sub: str, dependencies: dict, result_set: set):
    """
    Returns the set of the subtasks happening after `sub`.
    """

    def helper(sub):
        for subtask, deps in dependencies.items():
            if sub in deps:
                result_set.add(subtask)
                helper(subtask)

    helper(sub)
    return result_set


def get_dependees(sub: str, dependencies: dict, result_set: set):
    """
    Returns the set of the subtasks happening before `sub`.
    """
    dependencies = OrderedDict(sorted(dependencies.items(), reverse=True))

    def helper(sub):
        for subtask, deps in dependencies.items():
            if sub == subtask:
                if len(deps) > 0:
                    dep = deps[0]
                    result_set.add(dep)
                    helper(dep)

    helper(sub)
    return result_set


def get_subtasks_to_restart_with_dependencies(
    should_restart: dict, dependencies: dict, force_restart: str
):
    subtasks_to_restart = set([subtask for subtask, restart in should_restart.items() if restart])

    # The last task requires to be a dependency too
    last_subtask = next(reversed(dependencies))
    dependencies["_end"] = last_subtask

    deps_set = set()
    for sub_to_restart in subtasks_to_restart:
        deps_set = get_dependencies(sub_to_restart, dependencies, deps_set)

    should_restart_with_deps = subtasks_to_restart | deps_set
    if "_end" in should_restart_with_deps:
        should_restart_with_deps.remove("_end")

    # Adds the subtasks to force restart
    if len(force_restart) > 0:
        for subtask in force_restart.split(","):
            if subtask not in TASK_ARGS:
                raise ValueError(f"Invalid subtask name `{subtask}` given to `--force_restart`.")
            force_restart_deps = get_dependencies(subtask, dependencies, set())
            if "_end" in force_restart_deps:
                force_restart_deps.remove("_end")
            force_restart_deps.add(subtask)
            should_restart_with_deps = should_restart_with_deps | force_restart_deps

    should_restart_with_deps = {
        task: (task in should_restart_with_deps)
        for task in TASK_ARGS
    }

    return should_restart_with_deps


def flatten_arg_values(cfg):
    def helper(dict_or_val, flatten_list):
        if isinstance(dict_or_val, dict):
            for key, value in dict_or_val.items():
                if isinstance(value, dict):
                    helper(value, flatten_list)
                else:
                    helper(f"{key}={value}", flatten_list)
        else:
            flatten_list.append(dict_or_val)

    flatten_list = []
    helper(cfg, flatten_list)
    return flatten_list


def add_cfg_args_to_parser(cfg, parser):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v == "None":
            return None
        if v.lower() in ("true"):
            return True
        elif v.lower() in ("false"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def nested_dict_to_separator_dict(nested_dict, separator="."):
        def _create_separator_dict(x, key="", separator_dict={}, keys_to_ignore=[]):
            if isinstance(x, dict):
                for k, v in x.items():
                    kk = f"{key}{separator}{k}" if key else k
                    _create_separator_dict(x[k], kk, keys_to_ignore=keys_to_ignore)
            else:
                if not any([ignore in key for ignore in keys_to_ignore]):
                    separator_dict[key] = x
            return separator_dict

        return _create_separator_dict(deepcopy(nested_dict))

    separator_dict = nested_dict_to_separator_dict(cfg)

    for k, v in separator_dict.items():
        assert isinstance(v, Arg), f"Arguments should have type `Arg`, seen {type(v)} instead."
        v = v.type
        is_bool = v == type(True)
        dtype = str2bool if is_bool else v
        parser.add_argument(f"--{k}", type=dtype)

    return parser


def get_darpa_tc_node_feats_from_cfg(cfg):
    features = cfg.build_graphs.node_label_features
    return {
        "subject": list(map(lambda x: x.strip(), features.subject.split(","))),
        "file": list(map(lambda x: x.strip(), features.file.split(","))),
        "netflow": list(map(lambda x: x.strip(), features.netflow.split(","))),
    }


TASK_FINISHED_FILE = "done.txt"


def set_task_to_done(task_path: str):
    with open(os.path.join(task_path, TASK_FINISHED_FILE), "w") as f:
        f.write("Task done")
    print(f"Task done: {task_path}\n")


def get_days_from_cfg(cfg):
    if cfg._test_mode:
        # Get the day number of the first day in each set
        days = [
            int(days[0].split("_")[-1])
            for days in [cfg.dataset.train_files, cfg.dataset.val_files, cfg.dataset.test_files]
        ]
    else:
        days = list(
            map(
                lambda x: int(x.split("_")[1]),
                [*cfg.dataset.train_files, *cfg.dataset.val_files, *cfg.dataset.test_files],
            )
        )

    return days


def get_uncertainty_methods_to_run(cfg):
    yml_file = get_yml_file(os.path.join(UNCERTAINTY_EXP_YML_FOLDER, cfg._experiment))
    validate_yml_file(yml_file, TASK_ARGS)
    with open(yml_file, "r") as file:
        uncertainty_cfg = yaml.safe_load(file)
    methods = list(uncertainty_cfg["experiment"]["uncertainty"].keys())
    return methods


def decoder_matches_objective(decoder: str, objective: str):
    return not (
        (decoder in DECODERS_EDGE_LEVEL and objective in OBJECTIVES_NODE_LEVEL)
        or (decoder in DECODERS_NODE_LEVEL and objective in OBJECTIVES_EDGE_LEVEL)
    )
