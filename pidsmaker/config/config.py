# Datasets

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
}

# Arguments

TASK_DEPENDENCIES = {
    "build_graphs": [],
    "transformation": ["build_graphs"],
    "feat_training": ["transformation"],
    "feat_inference": ["feat_training"],
    "graph_preprocessing": ["feat_inference"],
    "gnn_training": ["graph_preprocessing"],
    "evaluation": ["gnn_training"],
    "tracing": ["evaluation"],
}

class AND(list):
    pass

class OR(list):
    pass

FEATURIZATIONS_CFG = {
    "word2vec": {
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
    "doc2vec": {
        "include_neighbors": bool,
        "epochs": int,
        "alpha": float,
    },
    "fasttext": {
        "min_count": int,
        "alpha": float,
        "window_size": int,
        "negative": int,
        "num_workers": int,
        "use_pretrained_fb_model": bool,
    },
    "alacarte": {
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
    "flash": {
        "min_count": int,
        "workers": int,
    },
    "provd": {
        "alpha": float,
        "k": int,
        "mpl": int,
        "n_time_windows": int,
        "n_neighbors": int,
        "contamination": float,
    },
    "hierarchical_hashing": {},
    "magic": {},
    "only_type": {},
    "only_ones": {},
}

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
    "sage": {
        "activation": str,
    },
    "GLSTM": {
        "in_dim": int,
        "out_dim": int,
    },
    "rcaid_gat": {},
    "magic_gat": {
        "num_layers": int,
        "num_heads": int,
        "negative_slope": float,
        "alpha_l": float,
        "activation": str,
    },
    "GLSTM": {},
    "GIN": {},
    "sum_aggregation": {},
    "custom_mlp": {
        "architecture_str": str,
    },
    "none": {},
}

DECODERS_NODE_LEVEL = ["node_mlp", "none", "magic_gat", "nodlink"]
DECODERS_EDGE_LEVEL = ["edge_mlp"]
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
    "nodlink": {},
    "none": {},
}

RECON_LOSSES = ["SCE", "MSE", "MSE_sum", "MAE", "none"]
PRED_LOSSES = ["cross_entropy", "BCE"]
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
]
OBJECTIVES = OBJECTIVES_NODE_LEVEL + OBJECTIVES_EDGE_LEVEL
OBJECTIVES_CFG = {
    # Prediction-based
    "predict_edge_type": {
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
        "balanced_loss": bool,
        "use_triplet_types": bool,
    },
    "predict_node_type": {
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
        "balanced_loss": bool,
    },
    "predict_masked_struct": {
        "loss": (str, OR(PRED_LOSSES)),
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
        "balanced_loss": bool,
    },
    "detect_edge_few_shot": {
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
    },
    "predict_edge_contrastive": {
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
        "inner_product": {
            "dropout": float,
        },
    },
    # Reconstruction-based
    "reconstruct_node_features": {
        "loss": (str, OR(RECON_LOSSES)),
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
    },
    "reconstruct_node_embeddings": {
        "loss": (str, OR(RECON_LOSSES)),
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
    },
    "reconstruct_edge_embeddings": {
        "loss": (str, OR(RECON_LOSSES)),
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
    },
    "reconstruct_masked_features": {
        "loss": (str, OR(RECON_LOSSES)),
        "mask_rate": float,
        "decoder": (str, OR(list(DECODERS_CFG.keys()))),
        **DECODERS_CFG,
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

THRESHOLD_METHODS = ["max_val_loss", "mean_val_loss", "threatrace", "magic", "flash", "nodlink"]

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
    "preprocessing": {
        "build_graphs": {
            "used_method": (str, OR(["orthrus", "magic"])),
            "use_all_files": bool,
            "mimicry_edge_num": int,
            "time_window_size": float,
            "use_hashed_label": bool,
            "fuse_edge": bool,
            "node_label_features": {
                "subject": (str, AND(["type", "path", "cmd_line"])),
                "file": (str, AND(["type", "path"])),
                "netflow": (str, AND(["type", "remote_ip", "remote_port"])),
            },
            "multi_dataset": (str, OR(list(DATASET_DEFAULT_CONFIG.keys()) + ["none"])),
        },
        "transformation": {
            "used_methods": (str, AND(["undirected", "dag", "rcaid_pseudo_graph", "none"] + list(SYNTHETIC_ATTACKS.keys()))),
            "rcaid_pseudo_graph": {
                "use_pruning": bool,
            },
            **SYNTHETIC_ATTACKS,
        },
    },
    "featurization": {
        "feat_training": {
            "emb_dim": int,
            "epochs": int,
            "use_seed": bool,
            "training_split": (str, OR(["train", "all"])),
            "multi_dataset_training": bool,
            "used_method": (str, OR(list(FEATURIZATIONS_CFG.keys()))),
            **FEATURIZATIONS_CFG,
        },
        "feat_inference": {
            "to_remove": bool,  # TODO: remove
        },
    },
    "detection": {
        "graph_preprocessing": {
            "node_features": (str, AND(["node_type", "node_emb", "only_ones", "edges_distribution"])),
            "edge_features": (str, AND(["edge_type", "edge_type_triplet", "msg", "time_encoding", "none"])),
            "multi_dataset_training": bool,
            "fix_buggy_graph_reindexer": bool,
            "global_batching": {
                "used_method": (str, OR(["edges", "minutes", "unique_edge_types", "none"])),
                "global_batching_batch_size": int,
                "global_batching_batch_size_inference": int,
            },
            "intra_graph_batching": {
                "used_methods": (str, OR(["edges", "tgn_last_neighbor", "none"])),
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
                "used_method": (str, OR(["graph_batching", "none"])),
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
            "used_method": (str, OR(["orthrus", "provd"])),
            "flash": {
                "in_channel": int,
                "out_channel": int,
                "lr": float,
                "weight_decay": float,
                "epochs": int,
            },
            "encoder": {
                "dropout": float,
                "used_methods": (str, AND(list(ENCODERS_CFG.keys()))),
                **ENCODERS_CFG,
            },
            "decoder": {
                "used_methods": (str, AND(list(OBJECTIVES_CFG.keys()))),
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
            "ground_truth_version": (str, OR(["orthrus"])),
            "best_model_selection": (str, OR(["best_adp"])),
            "used_method": str,
            "node_evaluation": {
                "threshold_method": (str, OR(THRESHOLD_METHODS)),
                "use_dst_node_loss": bool,
                "use_kmeans": bool,
                "kmeans_top_K": int,
            },
            "tw_evaluation": {
                "threshold_method": (str, OR(THRESHOLD_METHODS)),
            },
            "node_tw_evaluation": {
                "threshold_method": (str, OR(THRESHOLD_METHODS)),
                "use_dst_node_loss": bool,
                "use_kmeans": bool,
                "kmeans_top_K": int,
            },
            "queue_evaluation": {
                "queue_threshold": int,
                "used_method": (str, OR(["kairos_idf_queue", "provnet_lof_queue"])),
                "kairos_idf_queue": {
                    "include_test_set_in_IDF": bool,
                },
                "provnet_lof_queue": {
                    "queue_arg": str,
                },
            },
            "edge_evaluation": {
                "malicious_edge_selection": (str, OR(["src_node", "dst_node", "both_nodes"])),
                "threshold_method": (str, OR(THRESHOLD_METHODS)),
            },
        },
    },
    "triage": {
        "tracing": {
            "used_method": (str, OR(["depimpact"])),
            "depimpact": {
                "used_method": (str, OR(["component", "shortest_path", "1-hop", "2-hop", "3-hop"])),
                "score_method": (str, OR(["degree", "recon_loss", "degree_recon"])),
                "workers": int,
                "visualize": bool,
            },
        },
    },
    "postprocessing": {},
}

EXPERIMENTS_CONFIG = {
    "training_loop": {
        "run_evaluation": (str, OR(["each_epoch", "best_epoch"])), # (when to run inference on test set)
    },
    "experiment": {
        "used_method": (str, OR(["uncertainty", "none"])),
        "uncertainty": {
            "hyperparameter": {
                "hyperparameters": (str, AND(["lr, num_epochs, text_h_dim, gnn_h_dim"])),
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

