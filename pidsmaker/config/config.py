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


class Arg:
    def __init__(self, type, vals: list = None, desc: str = None):
        self.type = type
        self.vals = vals
        self.desc = desc


FEATURIZATIONS_CFG = {
    "word2vec": {
        "alpha": Arg(float),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "doc2vec": {
        "include_neighbors": Arg(bool),
        "epochs": Arg(int),
        "alpha": Arg(float),
    },
    "fasttext": {
        "min_count": Arg(int),
        "alpha": Arg(float),
        "window_size": Arg(int),
        "negative": Arg(int),
        "num_workers": Arg(int),
        "use_pretrained_fb_model": Arg(bool),
    },
    "alacarte": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "epochs": Arg(int),
        "context_window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "num_workers": Arg(int),
        "compute_loss": Arg(bool),
        "add_paths": Arg(bool),
    },
    "temporal_rw": {
        "walk_length": Arg(int),
        "num_walks": Arg(int),
        "trw_workers": Arg(int),
        "time_weight": Arg(str),
        "half_life": Arg(int),
        "window_size": Arg(int),
        "min_count": Arg(int),
        "use_skip_gram": Arg(bool),
        "wv_workers": Arg(int),
        "epochs": Arg(int),
        "compute_loss": Arg(bool),
        "negative": Arg(int),
        "decline_rate": Arg(int),
    },
    "flash": {
        "min_count": Arg(int),
        "workers": Arg(int),
    },
    "hierarchical_hashing": {},
    "magic": {},
    "only_type": {},
    "only_ones": {},
}

ENCODERS_CFG = {
    "tgn": {
        "tgn_memory_dim": Arg(int),
        "tgn_time_dim": Arg(int),
        "use_node_feats_in_gnn": Arg(bool),
        "use_memory": Arg(bool),
        "use_time_order_encoding": Arg(bool),
        "project_src_dst": Arg(bool),
    },
    "graph_attention": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "sage": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "gat": {
        "activation": Arg(str),
        "num_heads": Arg(int),
        "concat": Arg(bool),
        "flow": Arg(str),
        "num_layers": Arg(int),
    },
    "gin": {
        "activation": Arg(str),
        "num_layers": Arg(int),
    },
    "sum_aggregation": {},
    "rcaid_gat": {},
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "glstm": {},
    "custom_mlp": {
        "architecture_str": Arg(str),
    },
    "none": {},
}

DECODERS_NODE_LEVEL = ["node_mlp", "none", "magic_gat", "nodlink"]
DECODERS_EDGE_LEVEL = ["edge_mlp"]
DECODERS_CFG = {
    "edge_mlp": {
        "architecture_str": Arg(
            str,
            desc="A string describing a simple neural network. Example: if the encoder's output has shape `node_out_dim=128` \
                                setting `architecture_str=linear(2) | relu | linear(0.5)` creates this MLP: nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, y). \
                                Precisely, in linear(x), x is the multiplier of input neurons. The final layer `nn.Linear(128, y)` is added automatically such that `y` is the \
                                output size matching the downstream objective (e.g. edge type prediction involves predicting 10 edge types, so the output of the decoder should be 10).",
        ),
        "src_dst_projection_coef": Arg(
            int, desc="Multiplier of input neurons to project src and dst nodes."
        ),
    },
    "node_mlp": {
        "architecture_str": Arg(str),
    },
    "magic_gat": {
        "num_layers": Arg(int),
        "num_heads": Arg(int),
        "negative_slope": Arg(float),
        "alpha_l": Arg(float),
        "activation": Arg(str),
    },
    "nodlink": {},
    "inner_product": {},
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
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
        "use_triplet_types": Arg(bool),
    },
    "predict_node_type": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "predict_masked_struct": {
        "loss": Arg(str, vals=OR(PRED_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "balanced_loss": Arg(bool),
    },
    "detect_edge_few_shot": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "predict_edge_contrastive": {
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
        "inner_product": {
            "dropout": Arg(float),
        },
    },
    # Reconstruction-based
    "reconstruct_node_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_node_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_edge_embeddings": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
    "reconstruct_masked_features": {
        "loss": Arg(str, vals=OR(RECON_LOSSES)),
        "mask_rate": Arg(float),
        "decoder": Arg(
            str, vals=OR(list(DECODERS_CFG.keys())), desc="Decoder used before computing loss."
        ),
        **DECODERS_CFG,
    },
}

SYNTHETIC_ATTACKS = {
    "synthetic_attack_naive": {
        "num_attacks": Arg(int),
        "num_malicious_process": Arg(int),
        "num_unauthorized_file_access": Arg(int),
        "process_selection_method": Arg(str),
    },
}

THRESHOLD_METHODS = ["max_val_loss", "mean_val_loss", "threatrace", "magic", "flash", "nodlink"]

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
    "preprocessing": {
        "build_graphs": {
            "used_method": Arg(
                str, vals=OR(["default", "magic"]), desc="The method to build time window graphs."
            ),
            "use_all_files": Arg(bool),
            "mimicry_edge_num": Arg(int),
            "time_window_size": Arg(
                float,
                desc="The size of each graph in minutes. The notation should always be float (e.g. 10.0). Supports sizes < 1.0.",
            ),
            "use_hashed_label": Arg(bool, desc="Whether to hash the textual features."),
            "fuse_edge": Arg(
                bool, desc="Whether to fuse duplicate sequential edges into a single edge."
            ),
            "node_label_features": {
                "subject": Arg(
                    str,
                    vals=AND(["type", "path", "cmd_line"]),
                    desc="Which features use for process nodes. Features will be concatenated.",
                ),
                "file": Arg(
                    str,
                    vals=AND(["type", "path"]),
                    desc="Which features use for file nodes. Features will be concatenated.",
                ),
                "netflow": Arg(
                    str,
                    vals=AND(["type", "remote_ip", "remote_port"]),
                    desc="Which features use for netflow nodes. Features will be concatenated.",
                ),
            },
            "multi_dataset": Arg(
                str,
                vals=OR(list(DATASET_DEFAULT_CONFIG.keys()) + ["none"]),
                desc="A comma-separated list of datasets on which training is performed. Evaluation is done only the primary dataset run in CLI.",
            ),
        },
        "transformation": {
            "used_methods": Arg(
                str,
                vals=AND(
                    ["undirected", "dag", "rcaid_pseudo_graph", "none"]
                    + list(SYNTHETIC_ATTACKS.keys())
                ),
                desc="Applies transformations to graphs after their construction. Multiple transformations can be applied sequentially. Example: `used_methods=undirected,dag`",
            ),
            "rcaid_pseudo_graph": {
                "use_pruning": Arg(bool),
            },
            **SYNTHETIC_ATTACKS,
        },
    },
    "featurization": {
        "feat_training": {
            "emb_dim": Arg(
                int,
                desc="Size of the text embedding. Arg not used by some featurization methods that do not build embeddings.",
            ),
            "epochs": Arg(
                int, desc="Epochs to train the embedding method. Arg not used by some methods."
            ),
            "use_seed": Arg(bool),
            "training_split": Arg(
                str,
                vals=OR(["train", "all"]),
                desc="The partition of data used to train the featurization method.",
            ),
            "multi_dataset_training": Arg(
                bool,
                desc="Whether the featurization method should be trained on all datasets in `multi_dataset`.",
            ),
            "used_method": Arg(
                str,
                vals=OR(list(FEATURIZATIONS_CFG.keys())),
                desc="Algorithms used to create node and edge features.",
            ),
            **FEATURIZATIONS_CFG,
        },
        "feat_inference": {
            "to_remove": Arg(bool),  # TODO: remove
        },
    },
    "detection": {
        "graph_preprocessing": {
            "save_on_disk": Arg(
                bool,
                desc="Whether to store the graphs on disk upon building the graphs. \
                Used to avoid re-computation of very complex batching operations that take time. Can take up to 300GB storage for CADETS_E5.",
            ),
            "node_features": Arg(
                str,
                vals=AND(["node_type", "node_emb", "only_ones", "edges_distribution"]),
                desc="Node features to use during GNN training. `node_type` is a one-hot encoded entity type vector, \
                                     `node_emb` refers to the embedding generated during the `featurization` task, `only_ones` is a vector of ones \
                                      with length `node_type`, `edges_distribution` counts emitted and received edges.",
            ),
            "edge_features": Arg(
                str,
                vals=AND(["edge_type", "edge_type_triplet", "msg", "time_encoding", "none"]),
                desc="Edge features to used during GNN training. `edge_type` refers to the system call type, `edge_type_triplet` \
                                    considers a same edge type as a new type if source or destination node types are different, `msg` is the message vector \
                                    used in the TGN, `time_encoding` encodes temporal order of events with their timestamps in the TGN, `none` uses no features.",
            ),
            "multi_dataset_training": Arg(
                bool, desc="Whether the GNN should be trained on all datasets in `multi_dataset`."
            ),
            "fix_buggy_graph_reindexer": Arg(
                bool,
                desc="A bug has been found in the first version of the framework, where reindexing graphs in shape (N, d) \
                                                    slightly modify node features. Setting this to true fixes the bug.",
            ),
            "global_batching": {
                "used_method": Arg(
                    str,
                    vals=OR(["edges", "minutes", "unique_edge_types", "none"]),
                    desc="Flattens the time window-based graphs into a single large \
                                temporal graph and recreate graphs based on the given method. `edges` creates contiguous graphs of size `global_batching_batch_size` edges, \
                                the same applies for `minutes`, `unique_edge_types` builds graphs where each pair of connected nodes share edges with distinct edge types, \
                                `none` uses the default time window-based batching defined in minutes with arg `time_window_size`.",
                ),
                "global_batching_batch_size": Arg(
                    int,
                    desc="Controls the value associated with `global_batching.used_method` (training+inference).",
                ),
                "global_batching_batch_size_inference": Arg(
                    int,
                    desc="Controls the value associated with `global_batching.used_method` (inference only).",
                ),
            },
            "intra_graph_batching": {
                "used_methods": Arg(
                    str,
                    vals=AND(["edges", "tgn_last_neighbor", "none"]),
                    desc="Breaks each previously computed graph into even smaller graphs. \
                                    `edges` creates contiguous graphs of size `intra_graph_batch_size` edges (if a graph has 2000 edges and `intra_graph_batch_size=1500` \
                                    creates two graphs: one with 1500 edges, the other with 500 edges), `tgn_last_neighbor` computes for each graph its associated graph \
                                    based on the TGN last neighbor loader, namely a new graph where each node is connected with its last `tgn_neighbor_size` incoming edges.\
                                    `none` does not alter any graph.",
                ),
                "edges": {
                    "intra_graph_batch_size": Arg(
                        int,
                        desc="Controls the value associated with `global_batching.used_method`.",
                    ),
                },
                "tgn_last_neighbor": {
                    "tgn_neighbor_size": Arg(
                        int, desc="Number of last neighbors to store for each node."
                    ),
                    "tgn_neighbor_n_hop": Arg(
                        int,
                        desc="If greater than one, will also gather the last neighbors of neighbors.",
                    ),
                    "fix_buggy_orthrus_TGN": Arg(
                        bool,
                        desc="A bug has been in the first version of the framework, where the features of last neighbors not appearing \
                                                in the input graph have zero node feature vectors. Setting this arg to true includes the features of all nodes in the TGN graph.",
                    ),
                    "fix_tgn_neighbor_loader": Arg(
                        bool,
                        desc="We found a minor bug in the original TGN code (https://github.com/pyg-team/pytorch_geometric/issues/10100). This \
                                                    is an unofficial fix.",
                    ),
                    "directed": Arg(
                        bool,
                        desc="The original TGN's loader builds graphs in an undirected way. This makes the graphs purely directed.",
                    ),
                    "insert_neighbors_before": Arg(
                        bool,
                        desc="Whether to insert the edges of the current graph before loading last neighbors.",
                    ),
                },
            },
            "inter_graph_batching": {
                "used_method": Arg(
                    str,
                    vals=OR(["graph_batching", "none"]),
                    desc="Batches multiple graphs into a single large one for parallel training. \
                                    Does not support TGN. `graph_batching` batches `inter_graph_batch_size` together, `none` doesn't batch graphs.",
                ),
                "inter_graph_batch_size": Arg(
                    int,
                    desc="Controls the value associated with `inter_graph_batching.used_method`.",
                ),
            },
        },
        "gnn_training": {
            "use_seed": Arg(bool),
            "deterministic": Arg(
                bool, desc="Whether to force PyTorch to use deterministic algorithms."
            ),
            "num_epochs": Arg(int),
            "patience": Arg(int),
            "lr": Arg(float),
            "weight_decay": Arg(float),
            "node_hid_dim": Arg(int, desc="Number of neurons in the middle layers of the encoder."),
            "node_out_dim": Arg(int, desc="Number of neurons in the last layer of the encoder."),
            "grad_accumulation": Arg(
                int, desc="Number of epochs to gather gradients before backprop."
            ),
            "inference_device": Arg(
                str, vals=OR(["cpu", "cuda"]), desc="Device used during testing."
            ),
            "used_method": Arg(str, vals=OR(["default"]), desc="Which training pipeline use."),
            "encoder": {
                "dropout": Arg(float),
                "used_methods": Arg(
                    str,
                    vals=AND(list(ENCODERS_CFG.keys())),
                    desc="First part of the neural network. Usually GNN encoders to capture complex patterns.",
                ),
                "x_is_tuple": Arg(bool, desc="Whether to consider nodes differently when being source or destination."),
                **ENCODERS_CFG,
            },
            "decoder": {
                "used_methods": Arg(
                    str,
                    vals=AND(list(OBJECTIVES_CFG.keys())),
                    desc="Second part of the neural network. Usually MLPs specific to the downstream task (e.g. reconstruction of prediction)",
                ),
                **OBJECTIVES_CFG,
                "use_few_shot": Arg(bool, desc="Old feature: need some work to update it."),
                "few_shot": {
                    "include_attacks_in_ssl_training": Arg(bool),
                    "freeze_encoder": Arg(bool),
                    "num_epochs_few_shot": Arg(int),
                    "patience_few_shot": Arg(int),
                    "lr_few_shot": Arg(float),
                    "weight_decay_few_shot": Arg(float),
                    "decoder": {
                        "used_methods": Arg(str),
                        **OBJECTIVES_CFG,
                    },
                },
            },
        },
        "evaluation": {
            "viz_malicious_nodes": Arg(
                bool,
                desc="Whether to generate images of malicious nodes' neighborhoods (not stable).",
            ),
            "ground_truth_version": Arg(str, vals=OR(["orthrus", "reapr"])),
            "best_model_selection": Arg(
                str,
                vals=OR(["best_adp", "best_discrimination"]),
                desc="Strategy to select the best model across epochs. `best_adp` selects the best model based on the highest ADP score, `best_discrimination` \
                                        selects the model that does the best separation between top-score TPs and top-score FPs.",
            ),
            "used_method": Arg(str),
            "node_evaluation": {
                "threshold_method": Arg(
                    str,
                    vals=OR(THRESHOLD_METHODS),
                    desc="Method to calculate the threshold value used to detect anomalies.",
                ),
                "use_dst_node_loss": Arg(
                    bool,
                    desc="Whether to consider the loss of destination nodes when computing the node-level scores (maximum loss of a node).",
                ),
                "use_kmeans": Arg(
                    bool, desc="Whether to cluster nodes after thresholding as done in Orthrus"
                ),
                "kmeans_top_K": Arg(
                    int, desc="Number of top-score nodes selected before clustering."
                ),
            },
            "tw_evaluation": {
                "threshold_method": Arg(
                    str,
                    vals=OR(THRESHOLD_METHODS),
                    desc="Time-window detection. The code is broken and needs work to be updated.",
                ),
            },
            "node_tw_evaluation": {
                "threshold_method": Arg(
                    str,
                    vals=OR(THRESHOLD_METHODS),
                    desc="Node-level detection where a same node in multiple time windows is \
                        considered as multiple unique nodes. More realistic evaluation for near real-time detection. The code is broken and needs work to be updated.",
                ),
                "use_dst_node_loss": Arg(bool),
                "use_kmeans": Arg(bool),
                "kmeans_top_K": Arg(int),
            },
            "queue_evaluation": {
                "used_method": Arg(
                    str,
                    vals=OR(["kairos_idf_queue", "provnet_lof_queue"]),
                    desc="Queue-level detection as in Kairos. The code is broken and needs work to be updated.",
                ),
                "queue_threshold": Arg(int),
                "kairos_idf_queue": {
                    "include_test_set_in_IDF": Arg(bool),
                },
                "provnet_lof_queue": {
                    "queue_arg": Arg(str),
                },
            },
            "edge_evaluation": {
                "malicious_edge_selection": Arg(
                    str,
                    vals=OR(["src_node", "dst_node", "both_nodes"]),
                    desc="The ground truth only contains node-level labels. \
                    This arg controls the strategy to label edges. `src_nodes` and `dst_nodes` consider an edge as malicious if only its source or only its destination \
                    node is malicious. `both` labels an edge as malicious if both end nodes are malicious.",
                ),
                "threshold_method": Arg(str, vals=OR(THRESHOLD_METHODS)),
            },
        },
    },
    "triage": {
        "tracing": {
            "used_method": Arg(
                str,
                vals=OR(["depimpact"]),
                desc="Post-processing step to reconstruct attack paths or reduce false positives. `depimpact` is used in Orthrus.",
            ),
            "depimpact": {
                "used_method": Arg(
                    str, vals=OR(["component", "shortest_path", "1-hop", "2-hop", "3-hop"])
                ),
                "score_method": Arg(str, vals=OR(["degree", "recon_loss", "degree_recon"])),
                "workers": Arg(int),
                "visualize": Arg(bool),
            },
        },
    },
    "postprocessing": {},
}

EXPERIMENTS_CONFIG = {
    "training_loop": {
        "run_evaluation": Arg(
            str, vals=OR(["each_epoch", "best_epoch"])
        ),  # (when to run inference on test set)
    },
    "experiment": {
        "used_method": Arg(str, vals=OR(["uncertainty", "none"])),
        "uncertainty": {
            "hyperparameter": {
                "hyperparameters": Arg(str, vals=AND(["lr, num_epochs, text_h_dim, gnn_h_dim"])),
                "iterations": Arg(int),
                "delta": Arg(float),
            },
            "mc_dropout": {
                "iterations": Arg(int),
                "dropout": Arg(float),
            },
            "deep_ensemble": {
                "iterations": Arg(int),
                "restart_from": Arg(str),
            },
            "bagged_ensemble": {
                "iterations": Arg(int),
                "min_num_days": Arg(int),
            },
        },
    },
}
UNCERTAINTY_EXP_YML_FOLDER = "experiments/uncertainty/"
