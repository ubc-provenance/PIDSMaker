import argparse
import os
import hashlib
import pathlib
import sys
import yaml
import uuid
from copy import deepcopy
from collections import OrderedDict
from pprint import pprint
from yacs.config import CfgNode as CN


ROOT_ARTIFACT_DIR = "/home/artifacts/" # Destination folder (in the container) for generated files. Will be created if doesn't exist.
ROOT_GROUND_TRUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ground_Truth/")


DATABASE_DEFAULT_CONFIG = {
     "host": 'postgres',  # Host machine where the db is located
     "user": 'postgres',  # Database user
     "password": 'postgres',  # The password to the database user
     "port": '5432',  # The port number for Postgres
}
# ================================================================================

# --- Dependency graph to follow ---
TASK_DEPENDENCIES = OrderedDict({
     "build_graphs": [],
     "transformation": ["build_graphs"],
     "embed_nodes": ["transformation"],
     "embed_edges": ["embed_nodes"],
     "graph_preprocessing": ["embed_edges"],
     "gnn_training": ["graph_preprocessing"],
     "evaluation": ["gnn_training"],
     "tracing" : ["evaluation"],
})


ENCODERS = ["tgn", "graph_attention", "hetero_graph_transformer", "sage", "magic_gat", "custom_mlp", "none"]
ENCODERS_CFG = {
     "tgn": {
          "tgn_memory_dim": int,
          "tgn_time_dim": int,
          "use_node_feats_in_gnn": bool,
          "use_memory": bool,
          "use_time_order_encoding": bool,
          "project_src_dst": bool,
     },
     "graph_attention": {
          "activation": str,
          "num_heads": int,
          "concat": bool,
          "flow": str,
     },
     "hetero_graph_transformer": {
          "activation": str,
          "num_heads": int,
          "num_layers": int,
     },
     "sage": {
          "activation": str,
     },
     "LSTM":{
          "in_dim":int,
          "out_dim":int,
     },
     "magic_gat": {
          "num_layers": int,
          "num_heads": int,
          "negative_slope": float,
          "alpha_l": float,
          "activation": str,
     },
     "custom_mlp": {
          "architecture_str": str,
     },
}

DECODERS_NODE_LEVEL = ["node_mlp", "none", "magic_gat", "nodlink"]
DECODERS_EDGE_LEVEL = ["edge_mlp"]
DECODERS = DECODERS_NODE_LEVEL + DECODERS_EDGE_LEVEL
DECODERS_CFG = {
     "edge_mlp": {
          "architecture_str": str,
          "src_dst_projection_coef": int,
     },
     "node_mlp": {
          "architecture_str": str,
     },
     "magic_gat": {
          "num_layers": int,
          "num_heads": int,
          "negative_slope": float,
          "alpha_l": float,
          "activation": str,
     },
}

OBJECTIVES_NODE_LEVEL = ["predict_node_type", "reconstruct_node_features", "reconstruct_node_embeddings", "reconstruct_masked_features"]
OBJECTIVES_EDGE_LEVEL = ["predict_edge_type", "reconstruct_edge_embeddings", "predict_edge_contrastive", "predict_masked_struct"]
OBJECTIVES = OBJECTIVES_NODE_LEVEL + OBJECTIVES_EDGE_LEVEL
OBJECTIVES_CFG = {
     # Reconstruction-based
     "reconstruct_node_features": {
          "loss": str,  # ["SCE" | "MSE" | MSE_sum | MAE]
          "decoder": str,  # ["edge_mlp" | "custom_mlp" | "none"]
          **DECODERS_CFG,
     },
     "reconstruct_node_embeddings": {
          "loss": str,
          "decoder": str,
          **DECODERS_CFG,
     },
     "reconstruct_edge_embeddings": {
          "loss": str,
          "decoder": str,
          **DECODERS_CFG,
     },
     "reconstruct_masked_features":{
          "mask_rate": float,
          "loss": str,
          "decoder": str,
          **DECODERS_CFG,
     },
     # Prediction-based
     "predict_edge_type": {
          "decoder": str,
          **DECODERS_CFG,
          "balanced_loss": bool,
          "use_triplet_types": bool,
     },
     "predict_edge_type_hetero": {
          "decoder": str,
          **DECODERS_CFG,
          "balanced_loss": bool,
          "decoder_hetero_head": str,
     },
     "predict_node_type": {
          "decoder": str,
          **DECODERS_CFG,
          "balanced_loss": bool,
     },
     "predict_masked_struct":{
          "loss": str,
          "decoder": str,
          **DECODERS_CFG,
          "balanced_loss": bool,
     },
     "detect_edge_few_shot": {
          "decoder": str,
          **DECODERS_CFG,
     },
     "predict_edge_contrastive": {
          "decoder": str,
          **DECODERS_CFG,
          "inner_product": {
               "dropout": float,
          },
     },
}

SYNTHETIC_ATTACKS = {
     "synthetic_attack_naive": {
          "num_attacks": int,
          "num_malicious_process": int,
          "num_unauthorized_file_access": int,
          "process_selection_method": str,
     },
}

REQUIRE_HETERO_FEATURES_ENCODERS = ["hetero_graph_transformer"]
REQUIRE_NON_REVERSED_EDGES_ENCODERS = ["hetero_graph_transformer", "event_type_encoding"]

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
     "preprocessing": {
          "build_graphs": {
               "used_method": str, # [orthrus]
               "use_all_files": bool,
               "mimicry_edge_num": int,
               "time_window_size": float,
               "use_hashed_label": bool,
               "fuse_edge": bool,
               "node_label_features": {
                    "subject": str,  # [type, path, cmd_line]
                    "file": str,  # [type, path]
                    "netflow": str,  # [type, remote_ip, remote_port]
               },
               "multi_dataset": str,  # [CADETS_E3, THEIA_E3, ... | none]
          },
          "transformation": {
               "used_methods": str, # ["none", "rcaid_pseudo_graph", "undirected", "dag"]
               "rcaid_pseudo_graph": {
                    "use_pruning": bool,
               },
               **SYNTHETIC_ATTACKS,
          }
     },
     "featurization": {
          "embed_nodes": {
               "emb_dim": int,
               "epochs": int,
               "use_seed": bool,
               "training_split": str,  # ["train" | "all"]
               "multi_dataset_training": bool,
               "used_method": str,  # [ "temporal_rw" | "word2vec" | "doc2vec" | "feature_word2vec" | "hierarchical_hashing" | "only_type" | "flash" | "provd" | "fasttext"]
               "flash":{
                    "min_count": int,
                    "workers": int,
               },
               "temporal_rw": {
                    "walk_length": int,
                    "num_walks": int,
                    "trw_workers": int,
                    "time_weight": str,
                    "half_life": int,
                    'window_size': int,
                    'min_count': int,
                    'use_skip_gram': bool,
                    'wv_workers': int,
                    'epochs': int,
                    'compute_loss': bool,
                    'negative': int,
                    'decline_rate': int,
               },
               "word2vec": {
                    "walk_length": int,
                    "num_walks": int,
                    "epochs": int,
                    "context_window_size": int,
                    "min_count": int,
                    "use_skip_gram": bool,
                    "num_workers": int,
                    "compute_loss": bool,
                    "add_paths": bool,
               },
               "doc2vec": {
                    "include_neighbors": bool,
                    "epochs": int,
                    "alpha": float,
               },
               "feature_word2vec": {
                    "alpha": float,
                    'window_size': int,
                    'min_count': int,
                    'use_skip_gram': bool,
                    'num_workers': int,
                    'epochs': int,
                    'compute_loss': bool,
                    'negative': int,
                    'decline_rate': int,
               },
               "provd": {
                    "alpha": float,
                    "k": int,
                    "mpl": int,
                    "n_time_windows": int,
                    "n_neighbors": int,
                    "contamination": float,
               },
               "fasttext": {
                    "min_count": int,
                    "alpha": float,
                    "window_size": int,
                    "negative": int,
                    "num_workers": int,
                    "use_pretrained_fb_model": bool,
               },
          },
          "embed_edges": {
               "to_remove": bool, # TODO: remove
          }
     },
     "detection": {
          "graph_preprocessing": {
               "node_features": str,  # ["node_type", "node_emb", "edges_distribution"]
               "edge_features": str,  # ["edge_type", "edge_type_triplet", "msg", "time_encoding", "none"]
               "multi_dataset_training": bool,
               "fix_buggy_graph_reindexer": bool,
               "global_batching": {
                    "used_method": str,  # ["edges", "minutes", "unique_edge_types", "none"]
                    "global_batching_batch_size": int,  # [None | float] (if None, a batch is a TW of time_window_size) (training+inference)
                    "global_batching_batch_size_inference": int,  # [None | float] (only used during inference)
               },
               "intra_graph_batching": {
                    "used_methods": str,  # ["edges", "tgn_last_neighbor", "none"]
                    "edges": {
                         "intra_graph_batch_size": int,
                    },
                    "tgn_last_neighbor": {
                         "tgn_neighbor_size": int,
                         "tgn_neighbor_n_hop": int,
                         "fix_buggy_orthrus_TGN": bool,
                         "fix_tgn_neighbor_loader": bool,
                         "directed": bool,
                         "insert_neighbors_before": bool,
                    },
               },
               "inter_graph_batching": {
                    "used_method": str,  # ["graph_batching", "none"]
                    "inter_graph_batch_size": int,
               },
          },
          "gnn_training": {
               "use_seed": bool,
               "num_epochs": int,
               "patience": int,
               "lr": float,
               "weight_decay": float,
               "node_hid_dim": int,
               "node_out_dim": int,
               "grad_accumulation": int,
               "inference_device": str,
               "used_method": str, # [ "magic" | "orthrus" | "flash" ]
               "flash": {
                    "in_channel": int,
                    "out_channel": int,
                    "lr": float,
                    "weight_decay": float,
                    "epochs": int,
               },
               "encoder": {
                    "dropout": float,
                    "used_methods": str,  # [("graph_attention" | "sage" | "rcaid_gat" | "LSTM" | "custom_mlp" | "none"), "tgn", "ancestor_encoding", "entity_type_encoding", "event_type_encoding"]
                    **ENCODERS_CFG,
               },
               "decoder": {
                    "used_methods": str,  # ["reconstruct_edge_embeddings" | "predict_edge_type" | "predict_edge_contrastive" | "reconstruct_node_features" | "reconstruct_node_embeddings" | "predict_node_type"]
                    **OBJECTIVES_CFG,
                    "use_few_shot": bool,
                    "few_shot": {
                         "include_attacks_in_ssl_training": bool,
                         "freeze_encoder": bool,
                         "num_epochs_few_shot": int,
                         "patience_few_shot": int,
                         "lr_few_shot": float,
                         "weight_decay_few_shot": float,
                         "decoder": {
                              "used_methods": str,
                              **OBJECTIVES_CFG,
                         },
                    },
               },
          },
          "evaluation": {
               "viz_malicious_nodes": bool,
               "ground_truth_version": str,  # ["orthrus"]
               "best_model_selection": str,  # ["best_adp", "best_discrimination"]
               "used_method": str,
               "node_evaluation": {
                    "threshold_method": str,  # ["max_val_loss" | "mean_val_loss" | "threatrace" | "rcaid"]
                    "use_dst_node_loss": bool,
                    "use_kmeans": bool,
                    "kmeans_top_K": int,
               },
               "tw_evaluation": {
                    "threshold_method": str,  # ["max_val_loss" | "mean_val_loss"]
               },
               "node_tw_evaluation": {
                    "threshold_method": str,  # ["max_val_loss" | "mean_val_loss"]
                    "use_dst_node_loss": bool,
                    "use_kmeans": bool,
                    "kmeans_top_K": int,
               },
               "queue_evaluation": {
                    "queue_threshold": int,
                    "used_method": str,
                    "kairos_idf_queue": {
                         "include_test_set_in_IDF": bool,
                    },
                    "provnet_lof_queue": {
                         "queue_arg": str,
                    },
               },
               "edge_evaluation": {
                   "malicious_edge_selection": str,  # ["src_node" | "dst_node" | "both_nodes"]
                   "threshold_method": str,  # ["max_val_loss" | "mean_val_loss"]
               },
          },
     },
     "triage": {
          "tracing": {
               "used_method": str, #["depimpact"]
               "depimpact": {
                    "used_method": str, #["component" | "shortest_path" | "1-hop" | "2-hop" | "3-hop"]
                    "score_method": str, # ["degree" | "recon_loss" | "degree_recon"]
                    "workers": int,
                    "visualize" : bool,
               },
          },
     },
     "postprocessing":
          {},
}

EXPERIMENTS_CONFIG = {
     "training_loop": {
          "run_evaluation": str,  # ["each_epoch" | "best_epoch"] (when to run inference on test set)
     },
     "experiment": {
          "used_method": str,  # ["no_uncertainty" | "uncertainty"]
          "uncertainty": {
               "hyperparameter": {
                    "hyperparameters": str,  #  ["lr, num_epochs, text_h_dim, gnn_h_dim"]
                    "iterations": int,
                    "delta": float,
               },
               "mc_dropout": {
                    "iterations": int,
                    "dropout": float,
               },
               "deep_ensemble": {
                    "iterations": int,
                    "restart_from": str,
               },
               "bagged_ensemble": {
                    "iterations": int,
                    "min_num_days": int,
               }
          }
     }
}
UNCERTAINTY_EXP_YML_FOLDER = "experiments/uncertainty/"

DATASET_DEFAULT_CONFIG = {
     "THEIA_E5": {
          "raw_dir": "",
          "database": "theia_e5",
          "database_all_file": "theia_e5",
          # "database_all_file": "theia_e5_all", # NOTE: the whole dataset is too huge
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15"],
          "unused_files": ["graph_12", "graph_13", "graph_16", "graph_17"],
          "ground_truth_relative_path": ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"],
          "attack_to_time_window" : [
               ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv" , '2019-05-15 14:47:00', '2019-05-15 15:08:00'],
          ]
     },
     "THEIA_E3": {
          "raw_dir": "",
          "database": "theia_e3",
          "database_all_file": "theia_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_2", "graph_3", "graph_4", "graph_5", "graph_6", "graph_7", "graph_8"],
          "val_files": ["graph_9"],
          "test_files": ["graph_10", "graph_12", "graph_13"],
          "unused_files": ["graph_11"],
          "ground_truth_relative_path": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                                         "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                                         # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
                                         # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
                                         ],
          "attack_to_time_window" : [
               ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv" , '2018-04-12 12:40:00', '2018-04-12 13:30:00'],
               ["E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv" , '2018-04-10 14:30:00', '2018-04-10 15:00:00'],
          ]
     },
     "CADETS_E5": {
          "raw_dir": "",
          "database": "cadets_e5",
          "database_all_file": "cadets_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_11"],
          "val_files": ["graph_12"],
          "test_files": ["graph_16", "graph_17"],
          "unused_files": ["graph_15", "graph_10", "graph_13", "graph_14"],
          "ground_truth_relative_path": ["E5-CADETS/node_Nginx_Drakon_APT.csv",
                                         "E5-CADETS/node_Nginx_Drakon_APT_17.csv"],
          "attack_to_time_window" : [
               ["E5-CADETS/node_Nginx_Drakon_APT.csv" , '2019-05-16 09:31:00', '2019-05-16 10:12:00'],
               ["E5-CADETS/node_Nginx_Drakon_APT_17.csv" , '2019-05-17 10:15:00', '2019-05-17 15:33:00'],
          ]
     },
     "CADETS_E3": {
          "raw_dir": "",
          "database": "cadets_e3",
          "database_all_file": "cadets_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_2", "graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9"],
          "val_files": ["graph_10"],
          "test_files": ["graph_6", "graph_11", "graph_12", "graph_13"],
          "unused_files": [],
          "ground_truth_relative_path": [
                                         # "E3-CADETS/node_E_mail_Server.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_06.csv",
                                         # "E3-CADETS/node_Nginx_Backdoor_11.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_12.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_13.csv"],
          "attack_to_time_window": [
               ["E3-CADETS/node_Nginx_Backdoor_06.csv" , '2018-04-06 11:20:00', '2018-04-06 12:09:00'],
               # ["E3-CADETS/node_Nginx_Backdoor_11.csv" , '2018-04-11 15:07:00', '2018-04-11 15:16:00'],
               ["E3-CADETS/node_Nginx_Backdoor_12.csv" , '2018-04-12 13:59:00', '2018-04-12 14:39:00'],
               ["E3-CADETS/node_Nginx_Backdoor_13.csv" , '2018-04-13 09:03:00', '2018-04-13 09:16:00'],
          ],
     },
     "TRACE_E5": {
          "raw_dir": "",
          "database": "trace_e5",
          "database_all_file": "trace_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15"],
          "unused_files": ["graph_10", "graph_12", "graph_13", "graph_16", "graph_17"],
          "ground_truth_relative_path": [
               "E5-TRACE/node_Trace_Firefox_Drakon.csv",
          ],
          "attack_to_time_window": [
               ["E5-TRACE/node_Trace_Firefox_Drakon.csv", '2019-05-14 10:17:00', '2019-05-14 11:45:00'],
          ],
     },
     "TRACE_E3": {
          "raw_dir": "",
          "database": "trace_e3_all",
          "database_all_file": "trace_e3_all",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_6", "graph_11", "graph_12"],
          "val_files": ["graph_2"],
          "test_files": ["graph_13", "graph_10"],
          "unused_files": [],
          "ground_truth_relative_path": [
               "E3-TRACE/node_trace_e3_firefox_0410.csv",
               "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
               "E3-TRACE/node_trace_e3_pine_0413.csv",
               ],
          "attack_to_time_window": [
               ["E3-TRACE/node_trace_e3_firefox_0410.csv", '2018-04-10 09:45:00', '2018-04-10 11:10:00'],
               ["E3-TRACE/node_trace_e3_phishing_executable_0413.csv", '2018-04-13 14:14:00', '2018-04-13 14:29:00'],
               ["E3-TRACE/node_trace_e3_pine_0413.csv", '2018-04-13 12:42:00', '2018-04-13 12:54:00'],
          ],
     },
     "CLEARSCOPE_E5": {
          "raw_dir": "",
          "database": "clearscope_e5",
          "database_all_file": "clearscope_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_10", "graph_11", "graph_12"],
          "val_files": ["graph_13"],
          "test_files": ["graph_14", "graph_15", "graph_17"],
          "unused_files": ["graph_16"],
          "ground_truth_relative_path": [
               "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
               # "E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
          ],
          "attack_to_time_window": [
               ["E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", '2019-05-15 15:38:00', '2019-05-15 16:19:00'],
               # ["E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv", '2019-05-17 11:49:00', '2019-05-17 15:32:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv", '2019-05-17 15:48:00', '2019-05-17 16:01:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", '2019-05-17 16:20:00', '2019-05-17 16:28:00'],
          ],
     },
     "CLEARSCOPE_E3": {
          "raw_dir": "",
          "database": "clearscope_e3",
          "database_all_file": "clearscope_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_2"],
          "test_files": ["graph_11", "graph_12"],
          "unused_files": ["graph_6", "graph_13"],
          "ground_truth_relative_path": [
               "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
               # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
          ],
          "attack_to_time_window": [
               ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", '2018-04-11 13:54:00', '2018-04-11 14:48:00'],
               # ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", '2018-04-12 15:18:00', '2018-04-12 15:25:00'],
          ],
     },
     "FIVEDIRECTIONS_E5": {
          "raw_dir": "",
          "database": "fivedirections_e5_all",
          "database_all_file": "fivedirections_e5_all",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_10", "graph_11", "graph_13", "graph_14"],
          "val_files": ["graph_12"],
          "test_files": ["graph_15", "graph_17", "graph_9"],
          "unused_files": ["graph_16"],
          "ground_truth_relative_path": [
               "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
               "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
               "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
               "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
          ],
          "attack_to_time_window": [
               ["E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv", '2019-05-15 13:14:00', '2019-05-15 13:35:00'],
               ["E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv", '2019-05-09 13:25:00', '2019-05-09 13:57:00'],
               ["E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv", '2019-05-17 12:46:00', '2019-05-17 12:57:00'],
               ["E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv", '2019-05-17 16:10:00', '2019-05-17 16:16:00'],
          ],
     },
     "FIVEDIRECTIONS_E3": {
          "raw_dir": "",
          "database": "fivedirections_e3_all",
          "database_all_file": "fivedirections_e3_all",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_5", "graph_6", "graph_7", "graph_8", "graph_10", "graph_13"],
          "val_files": ["graph_4"],
          "test_files": ["graph_9", "graph_11"],
          "unused_files": ["graph_12"],
          "ground_truth_relative_path": [
               "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
               # "E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv",
               "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
          ],
          "attack_to_time_window": [
               ["E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv", '2018-04-11 09:59:00', '2018-04-11 10:41:00'],
               # ["E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv", '2018-04-12 11:12:00', '2018-04-12 11:15:00'],
               ["E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv", '2018-04-09 15:06:00', '2018-04-09 15:43:00']
          ],
     },
     "optc_h201": {
          "raw_dir": "",
          "database": "optc_201",
          "database_all_file": "optc_201",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-09",
          "start_end_day_range": (15, 26),
          "train_files": ["graph_19","graph_20","graph_21"],
          "val_files": ["graph_22"],
          "test_files": ["graph_23", "graph_24", "graph_25"],
          "unused_files": ["graph_16", "graph_17", "graph_18"],
          "ground_truth_relative_path": [
               "h201/node_h201_0923.csv",
          ],
          "attack_to_time_window": [
               ["h201/node_h201_0923.csv", "2019-09-23 11:23:00", "2019-09-23 13:25:00"],
          ],
     },
     "optc_h501": {
          "raw_dir": "",
          "database": "optc_501",
          "database_all_file": "optc_501",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-09",
          "start_end_day_range": (15, 26),
          "train_files": ["graph_19", "graph_20", "graph_21"],
          "val_files": ["graph_22"],
          "test_files": ["graph_23", "graph_24", "graph_25"],
          "unused_files": ["graph_16", "graph_17", "graph_18"],
          "ground_truth_relative_path": [
               "h501/node_h501_0924.csv",
          ],
          "attack_to_time_window": [
               ["h501/node_h501_0924.csv", "2019-09-24 10:28:00", "2019-09-24 15:29:00"],
          ],
     },
     "optc_h051": {
          "raw_dir": "",
          "database": "optc_051",
          "database_all_file": "optc_051",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-09",
          "start_end_day_range": (15, 26),
          "train_files": ["graph_19", "graph_20", "graph_21"],
          "val_files": ["graph_22"],
          "test_files": ["graph_23", "graph_24", "graph_25"],
          "unused_files": ["graph_16", "graph_17", "graph_18"],
          "ground_truth_relative_path": [
               "h051/node_h051_0925.csv",
          ],
          "attack_to_time_window": [
               ["h051/node_h051_0925.csv", "2019-09-25 10:29:00", "2019-09-25 14:25:00"],
          ],
     },
     "atlasv2_h1": {
          "raw_dir": "",
          "database": "cbc_edr_h1",
          "database_all_file": "cbc_edr_h1",
          "num_node_types": 3,
          "num_edge_types": 34,
          "year_month": "2022-07",
          "start_end_day_range": (14, 20),
          "train_files": ["graph_15", "graph_16",],
          "val_files": ["graph_17"],
          "test_files": ["graph_18","graph_19"],
          "unused_files": [],
          "ground_truth_relative_path": [
               "atlasv2_h1/node_h1_s1.csv",
               "atlasv2_h1/node_h1_s2.csv",
               "atlasv2_h1/node_h1_s3.csv",
               "atlasv2_h1/node_h1_s4.csv",
               "atlasv2_h1/node_h1_m1.csv",
               "atlasv2_h1/node_h1_m2.csv",
               "atlasv2_h1/node_h1_m3.csv",
               "atlasv2_h1/node_h1_m4.csv",
               "atlasv2_h1/node_h1_m5.csv",
               "atlasv2_h1/node_h1_m6.csv",
          ],
          "attack_to_time_window": [
               ["atlasv2_h1/node_h1_s1.csv", "2022-07-19 09:08:56", "2022-07-19 09:35:53"],
               ["atlasv2_h1/node_h1_s2.csv", "2022-07-19 09:43:11", "2022-07-19 10:16:35"],
               ["atlasv2_h1/node_h1_s3.csv", "2022-07-19 10:21:28", "2022-07-19 11:01:24"],
               ["atlasv2_h1/node_h1_s4.csv", "2022-07-19 20:31:58", "2022-07-19 21:04:44"],
               ["atlasv2_h1/node_h1_m1.csv", "2022-07-19 12:00:17", "2022-07-19 13:48:05"],
               ["atlasv2_h1/node_h1_m2.csv", "2022-07-19 15:27:58", "2022-07-19 16:02:07"],
               ["atlasv2_h1/node_h1_m3.csv", "2022-07-19 16:07:32", "2022-07-19 16:41:17"],
               ["atlasv2_h1/node_h1_m4.csv", "2022-07-19 18:32:12", "2022-07-19 19:05:03"],
               ["atlasv2_h1/node_h1_m5.csv", "2022-07-19 19:14:10", "2022-07-19 19:47:28"],
               ["atlasv2_h1/node_h1_m6.csv", "2022-07-19 19:54:13", "2022-07-19 20:26:49"],
          ],
     },
}

def get_default_cfg(args):
     """
     Inits the shared cfg object with default configurations.
     """
     cfg = CN()
     cfg._artifact_dir = ROOT_ARTIFACT_DIR

     cfg._test_mode = False
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
     for attr, value in DATABASE_DEFAULT_CONFIG.items():
          setattr(cfg.database, attr, value)

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
     parser.add_argument('model', type=str, help="Name of the model")
     parser.add_argument('dataset', type=str, help="Name of the dataset")
     parser.add_argument('--force_restart', type=str, default="", help="The subtask or subtasks from which to restart")
     parser.add_argument('--restart_from_scratch', action="store_true", help="Starts pipeline in a fresh new task path")
     parser.add_argument('--wandb', action="store_true", help="Whether to submit logs to wandb")
     parser.add_argument('--project', type=str, default="", help="Name of the wandb project (optional)")
     parser.add_argument('--exp', type=str, default="", help="Name of the experiment")
     parser.add_argument('--tags', type=str, default="", help="Name of the tag to use. Tags are used to group runs together")
     parser.add_argument('--cpu', action="store_true", help="Whether to run the framework on CPU rather than GPU")
     parser.add_argument('--experiment', type=str, default="no_experiment", help="The experiment yml config file")
     parser.add_argument('--tuning_mode', type=str, default="none", help="Name of the tuning mode to run the pipeline with wandb sweeps")
     parser.add_argument('--tuned', action="store_true", help="Whether to load the best fine-tuned hyperparameters")
     parser.add_argument('--tuning_file_path', default="", help="If set, use the given YML path for tuning")
     parser.add_argument('--sweep_id', default="", help="ID of a wandb sweep for multi-agent runs")

     # Script-specific args
     parser.add_argument('--show_attack', type=int, help="Number of attack for plotting", default=0)
     parser.add_argument('--gt_type', type=str, help="Type of ground truth", default="orthrus")
     parser.add_argument('--plot_gt', type=bool, help="If we plot ground truth", default=False)

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
     ```python benchmark.py --detection.gnn_training.seed=42```
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
     cfg._is_node_level = cfg.detection.gnn_training.decoder.used_methods is not None and \
          any([method for method in OBJECTIVES_NODE_LEVEL \
               if method in cfg.detection.gnn_training.decoder.used_methods])
          
     cfg._is_hetero = any([method for method in REQUIRE_HETERO_FEATURES_ENCODERS \
          if method in cfg.detection.gnn_training.encoder.used_methods])
     
     cfg._require_non_reversed_edges = any([method for method in REQUIRE_NON_REVERSED_EDGES_ENCODERS \
          if method in cfg.detection.gnn_training.encoder.used_methods])
     

def set_immutable_cfg(cfg):
     # In some hetero encoders, the reversed edges created by the TGN loader must be removed, or
     # invalid src, dst, edge triplets will exist
     if cfg._require_non_reversed_edges:
        tgn_loader_cfg = cfg.detection.graph_preprocessing.intra_graph_batching.tgn_last_neighbor
        tgn_loader_cfg.fix_buggy_orthrus_TGN = True
        tgn_loader_cfg.directed = True
        
     if "predict_edge_type_hetero" in cfg.detection.gnn_training.decoder.used_methods:
          cfg.detection.graph_preprocessing.edge_features = "edge_type"

def set_task_paths(cfg, subtask_concat_value=None):
     subtask_to_hash = {}
     # Directories common to all tasks
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)

          # We first compute a unique hash for each usbtask
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               restart_values = flatten_arg_values(subtask_cfg)
               if subtask_name == "build_graphs": # to restart from beginning if train files are changed
                    restart_values += cfg.dataset.train_files
                    if cfg._restart_from_scratch: # to start from a brand new folder, we generate a random id to add to the hash
                         restart_values += [cfg._run_random_seed]
               if subtask_concat_value is not None:
                    if subtask_name == subtask_concat_value["subtask"]:
                         restart_values += [subtask_concat_value["concat_value"]]

               clean_hash_args = ["".join([c for c in str(arg) if c not in set(" []\"\'")]) for arg in restart_values \
                    if not arg.startswith("_")]
               final_hash_string = ",".join(clean_hash_args)
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()
               
               subtask_to_hash[subtask_name] = final_hash_string

     # Then, for each subtask, we want its unique hash to also depend from its previous dependencies' hashes.
     # For example, if I run the same subtask A two times, with two different subtasks B and C, the results
     # would be different and would be stored in the same folder A if we don't consider the hash of B and C.
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               deps = sorted(list(get_dependees(subtask_name, TASK_DEPENDENCIES, set())))
               deps_hash = "".join([subtask_to_hash[dep] for dep in deps])
               
               final_hash_string = deps_hash + subtask_to_hash[subtask_name]
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()
               
               if task in ["preprocessing", "featurization"]:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, cfg.dataset.name, subtask_name, final_hash_string)
               else:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, subtask_name, final_hash_string, cfg.dataset.name)
               
               # The directory to save logs related to the preprocessing task
               subtask_cfg._logs_dir = os.path.join(subtask_cfg._task_path, "logs/")
               os.makedirs(subtask_cfg._logs_dir, exist_ok=True)
     
     # Preprocessing paths
     cfg.preprocessing.build_graphs._graphs_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "nx/")
     cfg.preprocessing.build_graphs._tw_labels = os.path.join(cfg.preprocessing.build_graphs._task_path, "tw_labels/")
     cfg.preprocessing.build_graphs._node_id_to_path = os.path.join(cfg.preprocessing.build_graphs._task_path, "node_id_to_path/")
     cfg.preprocessing.build_graphs._dicts_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "indexid2msg/")
     cfg.preprocessing.build_graphs._mimicry_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "mimicry/")
     cfg.preprocessing.build_graphs._magic_dir = os.path.join(cfg.preprocessing.build_graphs._task_path, "magic/")
     cfg.preprocessing.build_graphs._magic_graphs_dir = os.path.join(cfg.preprocessing.build_graphs._magic_dir, "dgl_graphs/")
     
     cfg.preprocessing.transformation._graphs_dir = os.path.join(cfg.preprocessing.transformation._task_path, "nx/")

     # Featurization paths
     cfg.featurization.embed_nodes._model_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "stored_models/")
     cfg.featurization.embed_nodes.temporal_rw._random_walk_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "random_walks/")
     cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir = os.path.join(cfg.featurization.embed_nodes.temporal_rw._random_walk_dir, "random_walk_corpus/")
     cfg.featurization.embed_nodes.word2vec._random_walk_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "random_walks/")
     cfg.featurization.embed_nodes.word2vec._random_walk_corpus_dir = os.path.join(cfg.featurization.embed_nodes.word2vec._random_walk_dir, "random_walk_corpus/")
     cfg.featurization.embed_nodes.word2vec._vec_graphs_dir = os.path.join(cfg.featurization.embed_nodes._task_path, "vectorized/")

     cfg.featurization.embed_edges._edge_embeds_dir = os.path.join(cfg.featurization.embed_edges._task_path, "edge_embeds/")
     cfg.featurization.embed_edges._model_dir = os.path.join(cfg.featurization.embed_edges._task_path, "stored_models/")


     # Detection paths
     cfg.detection.graph_preprocessing._preprocessed_graphs_dir = os.path.join(cfg.detection.graph_preprocessing._task_path, "preprocessed_graphs/")
     
     cfg.detection.gnn_training._trained_models_dir = os.path.join(cfg.detection.gnn_training._task_path, "trained_models/")
     cfg.detection.gnn_training._edge_losses_dir = os.path.join(cfg.detection.gnn_training._task_path, "edge_losses/")
     cfg.detection.gnn_training._magic_dir = os.path.join(cfg.detection.gnn_training._task_path, "magic/")
     cfg.detection.evaluation._precision_recall_dir = os.path.join(cfg.detection.evaluation._task_path, "precision_recall_dir/")
     cfg.detection.evaluation._uncertainty_exp_dir = os.path.join(cfg.detection.evaluation._task_path, "uncertainty_exp/")
     cfg.detection.evaluation.queue_evaluation._precision_recall_dir = os.path.join(cfg.detection.evaluation._task_path, "precision_recall_dir/")
     cfg.detection.evaluation.queue_evaluation._queues_dir = os.path.join(cfg.detection.evaluation._task_path, "queues_dir/")
     cfg.detection.evaluation.queue_evaluation._predicted_queues_dir = os.path.join(cfg.detection.evaluation._task_path, "predicted_queues_dir/")
     cfg.detection.evaluation.queue_evaluation._kairos_dir = os.path.join(cfg.detection.evaluation._task_path, "kairos_dir/")
     cfg.detection.evaluation._results_dir = os.path.join(cfg.detection.evaluation._task_path, "results/")

     # Ground Truth paths
     cfg._ground_truth_dir = os.path.join(ROOT_GROUND_TRUTH_DIR, cfg.detection.evaluation.ground_truth_version + '/')

     # Triage paths
     cfg.triage.tracing._tracing_graph_dir = os.path.join(cfg.triage.tracing._task_path, "tracing_graphs")

def validate_yml_file(yml_file: str, dictionary: dict):
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
     
     validate_config(user_config, dictionary)

def check_args(args):
     available_models = os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"))
     if not any([args.model in model for model in available_models]):
          raise ValueError(f"Unknown model {args.model}. Available models are {available_models}")
     
     available_datasets = DATASET_DEFAULT_CONFIG.keys()
     if args.dataset not in available_datasets:
          raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}")

def get_yml_file(filename, folder=""):
     root_path = pathlib.Path(__file__).parent.parent.resolve()
     return os.path.join(root_path, "config", folder, f"{filename}.yml")

def merge_cfg_and_check_syntax(cfg, yml_file, syntax_check=TASK_ARGS):
     user_config = CN(load_yml_file_recursive(yml_file, syntax_check=syntax_check))
     cfg.merge_from_other_cfg(user_config)
     return cfg

def load_yml_file_recursive(yml_file, syntax_check=TASK_ARGS):
     validate_yml_file(yml_file, syntax_check)
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file)
     if "_include_yml" in user_config:
          yml_to_include = get_yml_file(user_config["_include_yml"])
          included_config = load_yml_file_recursive(yml_to_include, syntax_check) or {}
          user_config = deep_merge_dicts(included_config, user_config)
     return user_config

def deep_merge_dicts(target, source):
    for k, v in source.items():
        target[k] = deep_merge_dicts(target[k], v) if isinstance(v, dict) and isinstance(target.get(k), dict) else v
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
          tuning_file = "orthrus" if args.model in ["orthrus_non_snooped", "orthrus_fixed"] else args.model
          if args.model == "orthrus_fixed" and args.dataset == "CLEARSCOPE_E3": # speciifc case
               tuning_file = args.model
          yml_file = get_yml_file(f"tuned_{tuning_file}", folder=f"tuned_baselines/{cfg.dataset.name.lower()}/")
          merge_cfg_and_check_syntax(cfg, yml_file)
     
     # Same for experiments
     yml_file = get_yml_file(os.path.join(UNCERTAINTY_EXP_YML_FOLDER, args.experiment))
     merge_cfg_and_check_syntax(cfg, yml_file, syntax_check=EXPERIMENTS_CONFIG)
     
     # Overwrites args to the cfg
     overwrite_cfg_with_args(cfg, args)
     
     # Here we create some variables based on parameters for easier usage
     set_shortcut_variables(cfg)
     
     # In some cases, we want some parameters to be fixed and unchanged
     set_immutable_cfg(cfg)

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
     decoders = cfg.detection.gnn_training.decoder.used_methods
     use_tgn_neigh_loader = "tgn_last_neighbor" in cfg.detection.graph_preprocessing.intra_graph_batching.used_methods
     use_tgn = "tgn" in cfg.detection.gnn_training.encoder.used_methods
     use_rcaid_pseudo_graph = "rcaid_pseudo_graph" in cfg.preprocessing.transformation.used_methods
     
     if use_tgn_neigh_loader:
          if use_rcaid_pseudo_graph:
               raise ValueError("Cannot use TGN with RCaid pseudo graph transformation. Edge timestamps are ignored with this transformation.")
          if not use_tgn:
               raise ValueError(f"Couldn't use `tgn_last_neighbor` without `tgn` as encoder.")
          
     if use_tgn:
          if not use_tgn_neigh_loader:
               raise ValueError(f"Couldn't use `tgn` as encoder without `tgn_last_neighbor` as loader.")
          
     if use_rcaid_pseudo_graph:
          if "predict_edge_type" in decoders:
               raise ValueError(f"Cannot predict edge type as it is removed in the pseudo graph transformation")
          
     if cfg.featurization.embed_nodes.used_method == "fasttext":
          if cfg.featurization.embed_nodes.fasttext.use_pretrained_fb_model:
               emb_dim = cfg.featurization.embed_nodes.emb_dim
               if emb_dim != 300:
                    raise ValueError(f"Invalid `emb_dim={emb_dim}`, should be set to 300 if `use_pretrained_fb_model=True`.")
               
     if "reconstruct_masked_features" in decoders or "predict_masked_struct" in decoders:
          if cfg.detection.evaluation.node_evaluation.threshold_method != "magic":
               raise ValueError(f"These decoders are only working with magic thresholding yet.")
          
     if cfg.detection.gnn_training.decoder.use_few_shot:
          if cfg.preprocessing.transformation.used_methods not in SYNTHETIC_ATTACKS.keys():
               raise ValueError("Few-shot mode requires an attack generation method within `preprocessing.transformation.used_methods`")
          
     use_multi_dataset = "none" not in cfg.preprocessing.build_graphs.multi_dataset
     if cfg.featurization.embed_nodes.multi_dataset_training and use_multi_dataset:
          method = cfg.featurization.embed_nodes.used_method.strip()
          if method not in ["feature_word2vec", "fasttext", "hierarchical_hashing", "only_type"]:
               raise NotImplementedError(f"Multi-dataset mode not implemented for method {method}")
     if cfg.featurization.embed_nodes.multi_dataset_training or cfg.detection.graph_preprocessing.multi_dataset_training:
          if not use_multi_dataset:
               raise ValueError(f"Using multi-dataset mode requires setting `preprocessing.build_graphs.multi_dataset`")
          
     if cfg.detection.evaluation.used_method == "edge_evaluation":
          if cfg._is_node_level:
               raise ValueError(f"Edge evaluation not implemented for node-level detection.")


def set_subtasks_to_restart(yml_file: str, cfg):
     """
     Given a cfg, returns a boolean for each subtask, being `True` if
     the subtask requires to be restarted and `False` if the current arguments
     do not require a restart.
     In practice, we restart a subtask if there is no TASK_FINISHED_FILE in its `_task_path`.
     """
     user_config = load_yml_file_recursive(yml_file)
     subtasks_in_yml_file = set([subtask for task, subtasks in user_config.items() if not task.startswith("_") for subtask in subtasks.keys()])

     should_restart = OrderedDict()
     for task, subtasks in TASK_ARGS.items():
          for subtask in subtasks.keys():
               if subtask in subtasks_in_yml_file:
                    subtask_cfg = getattr(getattr(cfg, task), subtask)
                    existing_files = [files for _, _, files in os.walk(subtask_cfg._task_path)]
                    has_finished = any([files for files in existing_files for f in files if f.endswith(TASK_FINISHED_FILE)])
                    should_restart[subtask] = not has_finished
               else:
                    should_restart[subtask] = False

     should_restart_with_deps = get_subtasks_to_restart_with_dependencies(should_restart, TASK_DEPENDENCIES, cfg._force_restart)
     
     # Dicts are not accepted in the cfg
     should_restart = [(subtask, restart) for subtask, restart in should_restart.items()]
     should_restart_with_deps = [(subtask, restart) for subtask, restart in should_restart_with_deps.items()]

     cfg._subtasks_should_restart = should_restart
     cfg._subtasks_should_restart_with_deps = should_restart_with_deps

def update_task_paths_to_restart(cfg, subtask_concat_value=None):
     """Simply recomputes if tasks should be restarted."""
     yml_file = get_yml_file(cfg._model)
     set_dataset_cfg(cfg, cfg.dataset.name)
     set_shortcut_variables(cfg)
     set_immutable_cfg(cfg)
     set_task_paths(cfg, subtask_concat_value=subtask_concat_value)
     set_subtasks_to_restart(yml_file, cfg)
     should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
     check_edge_cases(cfg)
     return should_restart

def update_cfg_for_multi_dataset(cfg, dataset):
    cfg = deepcopy(cfg)
    cfg.dataset.name = dataset
#     cfg.preprocessing.build_graphs.multi_dataset = "none"
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

def get_subtasks_to_restart_with_dependencies(should_restart: dict, dependencies: dict, force_restart: str):
     subtasks_to_restart = set([subtask for subtask, restart in should_restart.items() if restart])

     # The last task requires to be a dependency too
     last_subtask = next(reversed(dependencies))
     dependencies["_end"] = last_subtask
     
     deps_set = set()
     for sub_to_restart in subtasks_to_restart:
          deps_set = get_dependencies(sub_to_restart, dependencies, deps_set)

     should_restart_with_deps = (subtasks_to_restart | deps_set)
     if "_end" in should_restart_with_deps:
          should_restart_with_deps.remove("_end")

     # Adds the subtasks to force restart
     if len(force_restart) > 0:
          subtasks = set([subtask for task, subtasks in TASK_ARGS.items() for subtask in subtasks.keys()])
          for subtask in force_restart.split(","):
               if subtask not in subtasks:
                    raise ValueError(f"Invalid subtask name `{subtask}` given to `--force_restart`.")
               force_restart_deps = get_dependencies(subtask, dependencies, set())
               if "_end" in force_restart_deps:
                    force_restart_deps.remove("_end")
               force_restart_deps.add(subtask)
               should_restart_with_deps = (should_restart_with_deps | force_restart_deps)

     should_restart_with_deps = {subtask: (subtask in should_restart_with_deps) 
          for task, subtasks in TASK_ARGS.items()
          for subtask in subtasks.keys()}
     
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
          if v.lower() in ('true'):
               return True
          elif v.lower() in ('false'):
               return False
          else:
               raise argparse.ArgumentTypeError('Boolean value expected.')

     def nested_dict_to_separator_dict(nested_dict, separator='.'):
          def _create_separator_dict(x, key='', separator_dict={}, keys_to_ignore=[]):
               if isinstance(x, dict):
                    for k, v in x.items():
                         kk = f'{key}{separator}{k}' if key else k
                         _create_separator_dict(x[k], kk, keys_to_ignore=keys_to_ignore)
               else:
                    if not any([ignore in key for ignore in keys_to_ignore]):
                         separator_dict[key] = x
               return separator_dict

          return _create_separator_dict(deepcopy(nested_dict))
   
     separator_dict = nested_dict_to_separator_dict(cfg)

     for k, v in separator_dict.items():
          is_bool = v == type(True)
          dtype = str2bool if is_bool else v
          parser.add_argument(f'--{k}', type=dtype)

     return parser

def get_darpa_tc_node_feats_from_cfg(cfg):
    features = cfg.preprocessing.build_graphs.node_label_features
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
          days = [int(days[0].split("_")[-1]) for days in \
               [cfg.dataset.train_files, cfg.dataset.val_files, cfg.dataset.test_files]]
     else:
        days = list(map(lambda x: int(x.split("_")[1]), [*cfg.dataset.train_files, *cfg.dataset.val_files, *cfg.dataset.test_files]))
     
     return days

def get_uncertainty_methods_to_run(cfg):
     yml_file = get_yml_file(os.path.join(UNCERTAINTY_EXP_YML_FOLDER, cfg._experiment))
     validate_yml_file(yml_file, TASK_ARGS)
     with open(yml_file, 'r') as file:
          uncertainty_cfg = yaml.safe_load(file)
     methods = list(uncertainty_cfg["experiment"]["uncertainty"].keys())
     return methods

def decoder_matches_objective(decoder: str, objective: str):
     return not ((decoder in DECODERS_EDGE_LEVEL and objective in OBJECTIVES_NODE_LEVEL) or \
                 (decoder in DECODERS_NODE_LEVEL and objective in OBJECTIVES_EDGE_LEVEL))
