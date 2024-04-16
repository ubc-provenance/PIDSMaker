import argparse
import os
import pathlib
import sys
import yaml
from pprint import pprint
from yacs.config import CfgNode as CN


ROOT_ARTIFACT_DIR = "/data1/tbilot/artifact_new/"

AVAILABLE_MODELS = [
     "kairos_plus_plus",
]
AVAILABLE_DATASETS = [
     "THEIA_E5",
]

# --- Dependency graph to follow ---
TASK_DEPENDENCIES = {
     "build_graphs": [],
     "build_random_walks": ["build_graphs"],
     "embed_nodes": ["build_random_walks"],
}

# --- Tasks, subtasks, and argument configurations ---
# Restart args refer to the args which when modified requires to re-compute 
# the associated subtask and all subsequent subtasks
TASK_ARGS = {
     "preprocessing": {
          "build_graphs": {
               "time_window_size": int,
               "restart_args": ["time_window_size"],
     }},
     "featurization": {
          "build_random_walks": {
               "walk_length": int,
               "restart_args": ["walk_length"],
          },
          "embed_nodes": {
               "technique": str,  # ["alacarte"]
               "restart_args": ["technique"],
     }},
     "detection": {
          "model": {
               "name": str,
     }},
     "triage":
          {},
     "postprocessing":
          {},
}

DATASET_DEFAULT_CONFIG = {
     "THEIA_E5": {
          "max_node_num": 967390,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15"],
     }
}

def get_default_cfg(args):
     """
     Inits the shared cfg object with default configurations.
     """
     cfg = CN()
     cfg._artifact_dir = ROOT_ARTIFACT_DIR

     # Dataset: we simply create variables for all configurations described in the dict
     cfg.dataset = CN()
     cfg.dataset.name = args.dataset

     for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
          setattr(cfg.dataset, attr, value)
     
     # Tasks: we create nested None variables for all arguments
     for task, subtasks in TASK_ARGS.items():
          setattr(cfg, task, CN())
          task_cfg = getattr(cfg, task)

          for subtask, subtask_args in subtasks.items():
               setattr(task_cfg, subtask, CN())
               subtask_cfg = getattr(task_cfg, subtask)

               for arg, dtype in subtask_args.items():
                    setattr(subtask_cfg, arg, None)

     return cfg

def get_runtime_required_args():
     parser = argparse.ArgumentParser()
     parser.add_argument('model', type=str, help="Name of the model")
     parser.add_argument('dataset', type=str, help="Name of the dataset")
     
     try:
          args = parser.parse_args()
     except:
          parser.print_help()
          sys.exit(1)

     if args.model not in AVAILABLE_MODELS:
          raise ValueError(f"Unknown model {args.model}. Available models are {AVAILABLE_MODELS}")
     if args.dataset not in AVAILABLE_DATASETS:
          raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are {AVAILABLE_DATASETS}")

     return args

def set_task_paths(cfg):
     # Directories common to all tasks
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)

          # The directory where all subfolders of each task will be stored
          if task in ["preprocessing", "featurization"]:
               for subtask_name, subtask_args in subtask.items():
                    if "restart_args" not in subtask_args:
                         raise ValueError(f"Missing arg `restart_args` in subtask `{subtask_name}`")
                    restart_args = subtask_args["restart_args"]

                    subtask_cfg = getattr(task_cfg, subtask_name)
                    restart_values = [getattr(subtask_cfg, arg) for arg in restart_args]

                    clean_hash_args = "".join([c for c in str(restart_values) if c not in set(" []\"\'")])
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, cfg.dataset.name, subtask_name, clean_hash_args)

                    # The directory to save logs related to the preprocessing task
                    subtask_cfg._logs_dir = os.path.join(subtask_cfg._task_path, "logs/")
                    os.makedirs(subtask_cfg._logs_dir, exist_ok=True)
          else:
               hash_args = None # TODO set cfg.dataset.model for detection
     
     # Preprocessing paths
     # The directory to save the Networkx graphs
     cfg.preprocessing.build_graphs._graphs_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "nx/")

     # Featurization paths
     # The directory to save the preprocessed stuff from random walking
     cfg.featurization.build_random_walks._random_walk_dir = os.path.join(cfg.featurization.build_random_walks._task_path, "random_walks/")
     # The directory to save the preprocessed stuff from random walking
     cfg.featurization.build_random_walks._random_walk_corpus_dir = os.path.join(cfg.featurization.build_random_walks._random_walk_dir, "random_walk_corpus/")

     # The directory to save the vectorized graphs
     cfg.featurization.embed_nodes._vec_graphs_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "vectorized/")
     
     # TODO
     cfg.detection._task_path = None
     cfg.triage._task_path = None
     cfg.postprocessing._task_path = None

def validate_yml_file(yml_file: str):
     with open(yml_file, 'r') as file:
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
                         # Recursive check for sub-dictionaries
                         validate_config(sub_config, sub_tasks, path + [key])
                    else:
                         # Check for None values in parameters
                         if sub_config is None:
                              raise ValueError(f"Parameter '{' > '.join(path + [key])}' should not be None.")
                              # Optional: check for type correctness
                         if not isinstance(sub_config, sub_tasks):
                              raise TypeError(f"Parameter '{' > '.join(path + [key])}' should be of type {sub_tasks.__name__}.")
     
     validate_config(user_config, TASK_ARGS)
     print(f"YAML configuration file \"{yml_file.split('/')[-1]}\" is valid")

def check_task_dependency_graph(yml_file: str):
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file)
     
     subtasks = [j for i in user_config.values() for j in i]
     deps = TASK_DEPENDENCIES
     subtask_set = set(subtasks)

     def has_all_dependencies(task):
          return all(dependency in subtask_set and has_all_dependencies(dependency)
               for dependency in deps.get(task, []))

     dependencies_ok = all(has_all_dependencies(subtask) for subtask in subtasks)
     if dependencies_ok:
          print(f"Task dependency graph is valid: {subtasks}")
          print("\nYAML configuration")
          pprint(user_config)
     else:
          raise ValueError(("The requested subtasks don't respect the subtask dependency graph."
               f"Tasks: {subtasks}\nTask dependency graph: {deps}"))

def get_yml_cfg(args):
     # Inits with default configurations
     cfg = get_default_cfg(args)

     # Checks that all configurations are valid (not set to None)
     root_path = pathlib.Path(__file__).parent.parent.resolve()
     yml_file = f"{root_path}/config/{args.model}.yml"
     validate_yml_file(yml_file)

     # Overrides default config with config from yml file
     cfg.merge_from_file(yml_file)

     # Asserts all required configurations are present in the final cfg
     check_task_dependency_graph(yml_file)

     # Based on the defined restart args, computes a unique path on disk
     # to store the files of each task
     set_task_paths(cfg)

     print("\nFinal configuration:")
     pprint(dict(cfg.items()), indent=2)
     return cfg



########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = "/home/yinyuanl/Desktop/theia/"

# The directory to save all artifacts
artifact_dir = "./artifact/"

# The directory to save the Networkx graphs
graphs_dir = artifact_dir + "graphs/nx/"

# The directory to save the preprocessed stuff from random walking
preprocessed_dir = artifact_dir + "preprocessed/"

# The directory to save the vectorized graphs
vec_graphs_dir = artifact_dir + "graphs/vectorized/"

# The directory to save the GNN models
gnn_models_dir = artifact_dir + "gnn_models/"

# The directory to save the word2vec models
w2v_models_dir = artifact_dir + "w2v_models/"

# The directory to save the results after testing
test_re = artifact_dir + "test_re/"

# The directory to save all visualized results
vis_re = artifact_dir + "vis_re/"



########################################################
#
#               Database settings
#
########################################################

# Database name
database = 'theia_1_e5'

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
host = '/var/run/postgresql/'
# host = None

# Database user
user = 'postgres'

# The password to the database user
password = 'postgres'

# The port number for Postgres
port = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
     'EVENT_EXECUTE',
     'EVENT_LSEEK',
     'EVENT_MMAP',
     'EVENT_OPEN',
     'EVENT_ACCEPT',
     'EVENT_READ',
     'EVENT_RECVFROM',
     'EVENT_RECVMSG',
     'EVENT_READ_SOCKET_PARAMS',
     'EVENT_CHECK_FILE_ATTRIBUTES'
]

# The following edges are not considered to construct the
# temporal graph for experiments.
exclude_edge_type= [
     'EVENT_FCNTL',                          # EVENT_FCNTL does not have any predicate
     'EVENT_OTHER',                          # EVENT_OTHER does not have any predicate
     'EVENT_ADD_OBJECT_ATTRIBUTE',           # This is used to add attributes to an object that was incomplete at the time of publish
     'EVENT_FLOWS_TO',                       # No corresponding system call event
]

rel2id = {
        1: 'EVENT_CONNECT',
        'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE',
        'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN',
        'EVENT_OPEN': 3,
        4: 'EVENT_READ',
        'EVENT_READ': 4,
        5: 'EVENT_RECVFROM',
        'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG',
        'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG',
        'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO',
        'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE',
        'EVENT_WRITE': 9,
        10: 'EVENT_CLONE',
        'EVENT_CLONE': 10,
    }

########################################################
#
#                   Model dimensionality
#
########################################################

# Word Embedding Dimension
word_embedding_dim = 128

# Node Embedding Dimension
node_embedding_dim = word_embedding_dim

# Node State Dimension
node_state_dim = 100

# Neighborhood Sampling Size
neighbor_size = 20

# Graph Embedding Dimension
graph_dim = 64

# The time encoding Dimension
time_dim = 100


########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 1024

# Parameters for optimizer
lr=0.000001
eps=1e-08
weight_decay=0.01

epoch_num=50

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15


########################################################
#
#                   Threshold
#
########################################################

beta_day10 = 20
beta_day11 = 20