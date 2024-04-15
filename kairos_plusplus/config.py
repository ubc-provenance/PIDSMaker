import argparse
import os
import pathlib
import sys
from yacs.config import CfgNode as CN


AVAILABLE_MODELS = [
     "kairos_plus_plus",
]
AVAILABLE_DATASETS = [
     "THEIA_E5",
]

RESTART_ARGS = lambda cfg: {
     "preprocessing":
          {
               "THEIA_E5": [
                    # cfg.preprocessing.arg1,
                    # cfg.preprocessing.arg2,
               ]
          },
     "featurization": 
          {
               "THEIA_E5": [
                    # cfg.featurization.arg1,
                    # cfg.featurization.arg2,
               ]
          },
     "detection": 
          {
               "kairos_plus_plus": [
                    # cfg.detection.model.x_dim,
                    # cfg.detection.model.h_dim,
               ]
          }
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

     cfg._artifact_dir = "/data1/tbilot/artifact_new/"

     # Dataset
     cfg.dataset = CN()
     cfg.dataset.name = args.dataset

     for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
          setattr(cfg.dataset, attr, value)
     
     # Preprocessing
     cfg.preprocessing = CN()

     # Featurization
     cfg.featurization = CN()

     # Detection
     cfg.detection = CN()
     cfg.detection.model = CN()
     cfg.detection.model.name = args.model

     # Triage
     cfg.triage = CN()

     # Post-processing
     cfg.postprocessing = CN()

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
     restart_args = RESTART_ARGS(cfg)

     # Directories common to all tasks
     for task in ["preprocessing", "featurization", "detection", "triage", "postprocessing"]:
          task_cfg = getattr(cfg, task)

          # The directory where all subfolders of each task will be stored
          if task in ["preprocessing", "featurization"]:
               hash_args = str(restart_args[task][cfg.dataset.name])
               task_cfg._task_path = os.path.join(cfg._artifact_dir, task, cfg.dataset.name, hash_args)

               # The directory to save logs related to the preprocessing task
               task_cfg._logs_dir = os.path.join(task_cfg._task_path, "logs/")
               os.makedirs(task_cfg._logs_dir, exist_ok=True)
          else:
               hash_args = None # TODO set cfg.dataset.model for detection
     
     # Preprocessing paths
     # The directory to save the Networkx graphs
     cfg.preprocessing._graphs_dir = os.path.join(cfg.preprocessing._task_path, "nx/")
     os.makedirs(cfg.preprocessing._graphs_dir, exist_ok=True)
     # The directory to save the preprocessed stuff from random walking
     cfg.preprocessing._preprocessed_dir = os.path.join(cfg.preprocessing._task_path, "preprocessed/")
     os.makedirs(cfg.preprocessing._preprocessed_dir, exist_ok=True)

     # Featurization paths
     # The directory to save the vectorized graphs
     cfg.featurization._vec_graphs_dir = os.path.join(cfg.featurization._task_path, "vectorized/")
     os.makedirs(cfg.featurization._vec_graphs_dir, exist_ok=True)
     
     # TODO
     cfg.detection._task_path = None
     cfg.triage._task_path = None
     cfg.postprocessing._task_path = None

def assert_cfg_complete(cfg):
     pass
     # verify all is not None

def get_yml_cfg(args):
     # Inits with default configurations
     cfg = get_default_cfg(args)

     # Overrides default config with config from yml file
     root_path = pathlib.Path(__file__).parent.parent.resolve()
     yml_file = f"{root_path}/config/{cfg.detection.model.name}.yml"
     cfg.merge_from_file(yml_file)

     # Based on the defined restart args, computes a unique path on disk
     # to store the files of each task
     set_task_paths(cfg)

     # Asserts all required configurations are present in the final cfg
     assert_cfg_complete(cfg)
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