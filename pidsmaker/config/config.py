# --- Dependency graph to follow ---
TASK_DEPENDENCIES = {
    "build_graphs": [],
    "transformation": ["build_graphs"],
    "embed_nodes": ["transformation"],
    "embed_edges": ["embed_nodes"],
    "graph_preprocessing": ["embed_edges"],
    "gnn_training": ["graph_preprocessing"],
    "evaluation": ["gnn_training"],
    "tracing": ["evaluation"],
}


ENCODERS = [
    "tgn",
    "graph_attention",
    "hetero_graph_transformer",
    "sage",
    "magic_gat",
    "custom_mlp",
    "none",
]
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
    "LSTM": {
        "in_dim": int,
        "out_dim": int,
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

OBJECTIVES_NODE_LEVEL = [
    "predict_node_type",
    "reconstruct_node_features",
    "reconstruct_node_embeddings",
    "reconstruct_masked_features",
]
OBJECTIVES_EDGE_LEVEL = [
    "predict_edge_type",
    "reconstruct_edge_embeddings",
    "predict_edge_contrastive",
    "predict_masked_struct",
]
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
    "reconstruct_masked_features": {
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
    "predict_masked_struct": {
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
            "used_method": str,  # [orthrus]
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
            "used_methods": str,  # ["none", "rcaid_pseudo_graph", "undirected", "dag"]
            "rcaid_pseudo_graph": {
                "use_pruning": bool,
            },
            **SYNTHETIC_ATTACKS,
        },
    },
    "featurization": {
        "embed_nodes": {
            "emb_dim": int,
            "epochs": int,
            "use_seed": bool,
            "training_split": str,  # ["train" | "all"]
            "multi_dataset_training": bool,
            "used_method": str,  # [ "temporal_rw" | "word2vec" | "doc2vec" | "feature_word2vec" | "hierarchical_hashing" | "only_type" | "flash" | "provd" | "fasttext"]
            "flash": {
                "min_count": int,
                "workers": int,
            },
            "temporal_rw": {
                "walk_length": int,
                "num_walks": int,
                "trw_workers": int,
                "time_weight": str,
                "half_life": int,
                "window_size": int,
                "min_count": int,
                "use_skip_gram": bool,
                "wv_workers": int,
                "epochs": int,
                "compute_loss": bool,
                "negative": int,
                "decline_rate": int,
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
                "window_size": int,
                "min_count": int,
                "use_skip_gram": bool,
                "num_workers": int,
                "epochs": int,
                "compute_loss": bool,
                "negative": int,
                "decline_rate": int,
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
            "to_remove": bool,  # TODO: remove
        },
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
            "used_method": str,  # [ "magic" | "orthrus" | "flash" ]
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
            "used_method": str,  # ["depimpact"]
            "depimpact": {
                "used_method": str,  # ["component" | "shortest_path" | "1-hop" | "2-hop" | "3-hop"]
                "score_method": str,  # ["degree" | "recon_loss" | "degree_recon"]
                "workers": int,
                "visualize": bool,
            },
        },
    },
    "postprocessing": {},
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
            },
        },
    },
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
        "ground_truth_relative_path": [
            "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"
        ],
        "attack_to_time_window": [
            [
                "E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv",
                "2019-05-15 14:47:00",
                "2019-05-15 15:08:00",
            ],
        ],
    },
    "THEIA_E3": {
        "raw_dir": "",
        "database": "theia_e3",
        "database_all_file": "theia_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "year_month": "2018-04",
        "start_end_day_range": (2, 14),
        "train_files": [
            "graph_2",
            "graph_3",
            "graph_4",
            "graph_5",
            "graph_6",
            "graph_7",
            "graph_8",
        ],
        "val_files": ["graph_9"],
        "test_files": ["graph_10", "graph_12", "graph_13"],
        "unused_files": ["graph_11"],
        "ground_truth_relative_path": [
            "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
            "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
            # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
            # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
        ],
        "attack_to_time_window": [
            [
                "E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                "2018-04-12 12:40:00",
                "2018-04-12 13:30:00",
            ],
            [
                "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                "2018-04-10 14:30:00",
                "2018-04-10 15:00:00",
            ],
        ],
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
        "ground_truth_relative_path": [
            "E5-CADETS/node_Nginx_Drakon_APT.csv",
            "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
        ],
        "attack_to_time_window": [
            ["E5-CADETS/node_Nginx_Drakon_APT.csv", "2019-05-16 09:31:00", "2019-05-16 10:12:00"],
            [
                "E5-CADETS/node_Nginx_Drakon_APT_17.csv",
                "2019-05-17 10:15:00",
                "2019-05-17 15:33:00",
            ],
        ],
    },
    "CADETS_E3": {
        "raw_dir": "",
        "database": "cadets_e3",
        "database_all_file": "cadets_e3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "year_month": "2018-04",
        "start_end_day_range": (2, 14),
        "train_files": [
            "graph_2",
            "graph_3",
            "graph_4",
            "graph_5",
            "graph_7",
            "graph_8",
            "graph_9",
        ],
        "val_files": ["graph_10"],
        "test_files": ["graph_6", "graph_11", "graph_12", "graph_13"],
        "unused_files": [],
        "ground_truth_relative_path": [
            # "E3-CADETS/node_E_mail_Server.csv",
            "E3-CADETS/node_Nginx_Backdoor_06.csv",
            # "E3-CADETS/node_Nginx_Backdoor_11.csv",
            "E3-CADETS/node_Nginx_Backdoor_12.csv",
            "E3-CADETS/node_Nginx_Backdoor_13.csv",
        ],
        "attack_to_time_window": [
            ["E3-CADETS/node_Nginx_Backdoor_06.csv", "2018-04-06 11:20:00", "2018-04-06 12:09:00"],
            # ["E3-CADETS/node_Nginx_Backdoor_11.csv" , '2018-04-11 15:07:00', '2018-04-11 15:16:00'],
            ["E3-CADETS/node_Nginx_Backdoor_12.csv", "2018-04-12 13:59:00", "2018-04-12 14:39:00"],
            ["E3-CADETS/node_Nginx_Backdoor_13.csv", "2018-04-13 09:03:00", "2018-04-13 09:16:00"],
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
            [
                "E5-TRACE/node_Trace_Firefox_Drakon.csv",
                "2019-05-14 10:17:00",
                "2019-05-14 11:45:00",
            ],
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
        "train_files": [
            "graph_3",
            "graph_4",
            "graph_5",
            "graph_7",
            "graph_8",
            "graph_9",
            "graph_6",
            "graph_11",
            "graph_12",
        ],
        "val_files": ["graph_2"],
        "test_files": ["graph_13", "graph_10"],
        "unused_files": [],
        "ground_truth_relative_path": [
            "E3-TRACE/node_trace_e3_firefox_0410.csv",
            "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
            "E3-TRACE/node_trace_e3_pine_0413.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-TRACE/node_trace_e3_firefox_0410.csv",
                "2018-04-10 09:45:00",
                "2018-04-10 11:10:00",
            ],
            [
                "E3-TRACE/node_trace_e3_phishing_executable_0413.csv",
                "2018-04-13 14:14:00",
                "2018-04-13 14:29:00",
            ],
            ["E3-TRACE/node_trace_e3_pine_0413.csv", "2018-04-13 12:42:00", "2018-04-13 12:54:00"],
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
            [
                "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
                "2019-05-15 15:38:00",
                "2019-05-15 16:19:00",
            ],
            # ["E5-CLEARSCOPE/node_clearscope_e5_firefox_0517.csv", '2019-05-17 11:49:00', '2019-05-17 15:32:00'],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
                "2019-05-17 15:48:00",
                "2019-05-17 16:01:00",
            ],
            [
                "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
                "2019-05-17 16:20:00",
                "2019-05-17 16:28:00",
            ],
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
        "train_files": [
            "graph_3",
            "graph_4",
            "graph_5",
            "graph_7",
            "graph_8",
            "graph_9",
            "graph_10",
        ],
        "val_files": ["graph_2"],
        "test_files": ["graph_11", "graph_12"],
        "unused_files": ["graph_6", "graph_13"],
        "ground_truth_relative_path": [
            "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
            # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
        ],
        "attack_to_time_window": [
            [
                "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
                "2018-04-11 13:54:00",
                "2018-04-11 14:48:00",
            ],
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
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_bits_0515.csv",
                "2019-05-15 13:14:00",
                "2019-05-15 13:35:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_copykatz_0509.csv",
                "2019-05-09 13:25:00",
                "2019-05-09 13:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_dns_0517.csv",
                "2019-05-17 12:46:00",
                "2019-05-17 12:57:00",
            ],
            [
                "E5-FIVEDIRECTIONS/node_fivedirections_e5_drakon_0517.csv",
                "2019-05-17 16:10:00",
                "2019-05-17 16:16:00",
            ],
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
        "train_files": [
            "graph_3",
            "graph_5",
            "graph_6",
            "graph_7",
            "graph_8",
            "graph_10",
            "graph_13",
        ],
        "val_files": ["graph_4"],
        "test_files": ["graph_9", "graph_11"],
        "unused_files": ["graph_12"],
        "ground_truth_relative_path": [
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
            # "E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv",
            "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
        ],
        "attack_to_time_window": [
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_firefox_0411.csv",
                "2018-04-11 09:59:00",
                "2018-04-11 10:41:00",
            ],
            # ["E3-FIVEDIRECTIONS/node_fivedirections_e3_browser_0412.csv", '2018-04-12 11:12:00', '2018-04-12 11:15:00'],
            [
                "E3-FIVEDIRECTIONS/node_fivedirections_e3_excel_0409.csv",
                "2018-04-09 15:06:00",
                "2018-04-09 15:43:00",
            ],
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
        "train_files": ["graph_19", "graph_20", "graph_21"],
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
        "train_files": [
            "graph_15",
            "graph_16",
        ],
        "val_files": ["graph_17"],
        "test_files": ["graph_18", "graph_19"],
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
