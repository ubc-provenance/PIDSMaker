import os
from itertools import product

import pytest
import torch
import wandb

from pidsmaker import main
from pidsmaker.config import (
    ENCODERS_CFG,
    get_runtime_required_args,
    get_yml_cfg,
)

# Set by `framework_setup_teardown` (see the `artifact_dir` fixture of conftest.py)
TESTS_ARTIFACT_DIR = None


def prepare_cfg(
    model,
    dataset,
    device="cuda",
    transformation=None,
    featurization=None,
    encoder=None,
    objective=None,
    decoder=None,
    custom_args=None,
    save_figures=False,
):
    input_args = [model, dataset]
    args, _ = get_runtime_required_args(return_unknown_args=True, args=input_args)
    args.__dict__["artifact_dir"] = TESTS_ARTIFACT_DIR

    if transformation:
        args.__dict__["transformation.used_methods"] = transformation
    if featurization:
        args.__dict__["featurization.used_method"] = featurization
    if encoder:
        args.__dict__["training.encoder.used_methods"] = encoder
    if objective:
        args.__dict__["training.decoder.used_methods"] = objective
    if decoder and objective:
        args.__dict__[f"training.decoder.{objective}.decoder"] = decoder

    if encoder and "tgn" not in encoder:
        args.__dict__["batching.intra_graph_batching.used_methods"] = "edges"

    # Plotting dominates the evaluation time: figures are only tested by `TestEvaluationFigures`
    args.__dict__["evaluation.save_figures"] = save_figures

    if custom_args:
        for k, v in custom_args:
            args.__dict__[k] = v

    # The node type is the label of `predict_node_type`: it cannot be a node feature
    objectives = args.__dict__.get("training.decoder.used_methods") or ""
    if "predict_node_type" in objectives and args.__dict__.get("batching.node_features") is None:
        args.__dict__["batching.node_features"] = "node_emb"

    cfg = get_yml_cfg(args)
    cfg._test_mode = True

    if device == "cpu":
        cfg._use_cpu = True

    return cfg


@pytest.fixture(scope="session", autouse=True)
def framework_setup_teardown(artifact_dir):
    global TESTS_ARTIFACT_DIR
    TESTS_ARTIFACT_DIR = artifact_dir
    wandb.init(mode="disabled")
    yield


@pytest.fixture(scope="class")
def dataset():
    return "CLEARSCOPE_E3"


@pytest.fixture(scope="session")
def device(pytestconfig):
    return pytestconfig.getoption("device")


class TestTransformation:
    transformations = [
        "none",
        "rcaid_pseudo_graph",
        "undirected",
        "dag",
    ]
    failing_with_tgn = [
        "rcaid_pseudo_graph",
    ]

    @pytest.mark.parametrize("transformation", transformations)
    def test_transformations_tgn(self, dataset, transformation):
        if transformation in self.failing_with_tgn:
            with pytest.raises(ValueError):
                cfg = prepare_cfg("tests", dataset, transformation=transformation)
                main.main(cfg)
        else:
            cfg = prepare_cfg("tests", dataset, transformation=transformation)
            main.main(cfg)

    @pytest.mark.parametrize("transformation", transformations)
    def test_transformations(self, dataset, transformation):
        cfg = prepare_cfg("nodlink", dataset, transformation=transformation)
        main.main(cfg)


class TestFeaturization:
    featurizations = [
        "word2vec",
        "doc2vec",
        "hierarchical_hashing",
        "only_type",
        "only_ones",
        "fasttext",
        "flash",
        "alacarte",
        "temporal_rw",
        "graph2vec",
    ]

    @pytest.mark.parametrize("featurization", featurizations)
    def test_featurizations(self, dataset, featurization):
        cfg = prepare_cfg("tests", dataset, featurization=featurization)
        main.main(cfg)


class TestTokenization:
    # Featurization methods combined with the tokenization of other systems.
    # `huggingface` is not tested as it downloads a tokenizer from HuggingFace.
    pairs = [
        ("word2vec", "nltk"),
        ("word2vec", "hierarchical"),
        ("flash", "whole_label"),
        ("word2vec", "bpe"),
        ("doc2vec", "bpe"),
        ("fasttext", "hierarchical"),
        ("flash", "bpe"),
        ("hierarchical_hashing", "nltk"),
    ]

    @pytest.mark.parametrize("featurization,tokenization", pairs)
    def test_tokenizations(self, dataset, featurization, tokenization):
        custom_args = [
            ("featurization.tokenization.used_method", tokenization),
            ("featurization.tokenization.nltk.type_aware", False),
        ]
        cfg = prepare_cfg("tests", dataset, featurization=featurization, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize(
        "model,featurization,tokenization",
        [
            ("tests", "word2vec", "none"),  # word2vec needs tokens
            ("pretrained_velox", None, "nltk"),  # spider is built around BPE
        ],
    )
    def test_invalid_tokenizations(self, dataset, model, featurization, tokenization):
        custom_args = [("featurization.tokenization.used_method", tokenization)]
        with pytest.raises(ValueError, match="tokenization"):
            prepare_cfg(model, dataset, featurization=featurization, custom_args=custom_args)

    @pytest.mark.parametrize(
        "featurization,tokenization,substitutions",
        [
            ("word2vec", "nltk", "hash, guid, random, number, ip, port"),
            ("hierarchical_hashing", "hierarchical", "hash, random, number"),
            ("word2vec", "bpe", "none"),  # BPE without any placeholder
            ("doc2vec", "bpe", "hash, guid, ip"),
        ],
    )
    def test_substitutions(self, dataset, featurization, tokenization, substitutions):
        custom_args = [
            ("featurization.tokenization.used_method", tokenization),
            (f"featurization.tokenization.{tokenization}.substitutions", substitutions),
        ]
        cfg = prepare_cfg("tests", dataset, featurization=featurization, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("substitutions", ["hash, uuid", "none, hash", ""])
    def test_invalid_substitutions(self, dataset, substitutions):
        custom_args = [("featurization.tokenization.nltk.substitutions", substitutions)]
        with pytest.raises(ValueError, match="substitutions"):
            prepare_cfg("tests", dataset, custom_args=custom_args)


class TestCleaning:
    methods = [
        "remove_null_attributes",
        "normalize_labels",
        "remove_duplicate_events",
        "remove_self_loops",
        "remove_noisy_netflows",
        "dedup_window_edges",
    ]

    @pytest.mark.parametrize("cleaning", methods + [",".join(methods)])
    def test_cleaning(self, dataset, cleaning):
        custom_args = [("construction.cleaning.used_methods", cleaning)]
        cfg = prepare_cfg("tests", dataset, custom_args=custom_args)
        main.main(cfg)
        assert os.path.exists(os.path.join(cfg.construction._task_path, "cleaning_report.json"))

    def test_cleaning_options(self, dataset):
        custom_args = [
            ("construction.cleaning.used_methods", "normalize_labels, remove_noisy_netflows"),
            ("construction.cleaning.normalize_labels.lowercase", True),
            ("construction.cleaning.normalize_labels.max_length", 0),
            (
                "construction.cleaning.remove_noisy_netflows.ip_categories",
                "multicast, loopback, unspecified",
            ),
        ]
        cfg = prepare_cfg("tests", dataset, custom_args=custom_args)
        main.main(cfg)

    def test_no_cleaning_keeps_hash(self, dataset):
        # Graphs built before cleaning existed are reused
        cfg = prepare_cfg("tests", dataset)
        cfg_none = prepare_cfg(
            "tests", dataset, custom_args=[("construction.cleaning.used_methods", "none")]
        )
        cfg_clean = prepare_cfg(
            "tests",
            dataset,
            custom_args=[("construction.cleaning.used_methods", "remove_self_loops")],
        )
        assert cfg.construction._task_path == cfg_none.construction._task_path
        assert cfg.construction._task_path != cfg_clean.construction._task_path

    @pytest.mark.parametrize(
        "model,custom_args",
        [
            ("tests", [("construction.cleaning.used_methods", "remove_self_loops, typo")]),
            # kairos.yml does not include default.yml, which holds the normalize_labels args
            ("kairos", [("construction.cleaning.used_methods", "normalize_labels")]),
        ],
    )
    def test_invalid_cleaning(self, dataset, model, custom_args):
        with pytest.raises(ValueError, match="cleaning"):
            prepare_cfg(model, dataset, custom_args=custom_args)


class TestEncoderObjective:
    # `tgn` is a wrapper, covered by test_encoder_tgn_objective_pairs below. `node_id_embedding`
    # and `snapshot_gru` wrap another encoder too, covered by test_snapshot_wrappers_objective_pairs.
    # `textrcnn` embeds snapshots, not nodes: it is covered by TestProGrapher.
    encoders = [
        e
        for e in ENCODERS_CFG.keys()
        if e not in ("tgn", "node_id_embedding", "snapshot_gru", "textrcnn")
    ]
    objectives = [
        "predict_node_type",
        "reconstruct_node_embeddings",
        "reconstruct_node_features",
        "predict_edge_type",
        "reconstruct_edge_embeddings",
        "predict_edge_contrastive",
    ]

    @pytest.mark.parametrize("encoder,objective", list(product(encoders, objectives)))
    def test_encoder_objective_pairs(self, dataset, device, encoder, objective):
        cfg = prepare_cfg("tests", dataset, device=device, encoder=encoder, objective=objective)
        main.main(cfg)

    @pytest.mark.parametrize("encoder,objective", list(product(encoders, objectives)))
    def test_encoder_tgn_objective_pairs(self, dataset, device, encoder, objective):
        encoder_combined = f"{encoder},tgn"
        # Per-type encoders need node types aligned with the TGN neighborhood, which only the fixed TGN batching gives
        custom_args = None
        if encoder == "rgcn_per_type":
            custom_args = [("batching.intra_graph_batching.tgn_last_neighbor.fix_buggy_orthrus_TGN", True)]
        cfg = prepare_cfg(
            "tests", dataset, device=device, encoder=encoder_combined, objective=objective, custom_args=custom_args
        )
        main.main(cfg)

    @pytest.mark.parametrize("objective", objectives)
    def test_snapshot_wrappers_objective_pairs(self, dataset, device, objective):
        # Encoder of Euler: GCN over node ID embeddings, followed by a GRU over the graphs
        encoder = "gcn,node_id_embedding,snapshot_gru"
        cfg = prepare_cfg("tests", dataset, device=device, encoder=encoder, objective=objective)
        main.main(cfg)


class TestDecoderObjective:
    node_decoders = [
        "node_mlp",
    ]
    edge_decoders = [
        "edge_mlp",
    ]
    node_level_objectives = [
        "predict_node_type",
        "reconstruct_node_embeddings",
        "reconstruct_node_features",
    ]
    edge_level_objectives = [
        "predict_edge_type",
        "reconstruct_edge_embeddings",
        "predict_edge_contrastive",
    ]

    @pytest.mark.parametrize(
        "decoder,objective", list(product(node_decoders, node_level_objectives))
    )
    def test_decoder_objective_pairs_node_level_success(self, dataset, device, decoder, objective):
        cfg = prepare_cfg("tests", dataset, device=device, decoder=decoder, objective=objective)
        main.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(edge_decoders, edge_level_objectives))
    )
    def test_decoder_objective_pairs_edge_level_success(self, dataset, device, decoder, objective):
        cfg = prepare_cfg("tests", dataset, device=device, decoder=decoder, objective=objective)
        main.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(node_decoders, edge_level_objectives))
    )
    def test_decoder_objective_pairs_node_level_fail(self, dataset, device, decoder, objective):
        with pytest.raises(ValueError):
            cfg = prepare_cfg("tests", dataset, device=device, decoder=decoder, objective=objective)
            main.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(edge_decoders, node_level_objectives))
    )
    def test_decoder_objective_pairs_edge_level_fail(self, dataset, device, decoder, objective):
        with pytest.raises(ValueError):
            cfg = prepare_cfg("tests", dataset, device=device, decoder=decoder, objective=objective)
            main.main(cfg)


class TestBatching:
    global_batching_methods = [
        "edges",
        "minutes",
        "unique_edge_types",
        "none",
    ]
    intra_graph_batching_methods = [
        "edges",
        "tgn_last_neighbor",
        "edges,tgn_last_neighbor",
        "none",
    ]
    inter_graph_batching_methods = [
        "graph_batching",
        "none",
    ]

    @pytest.mark.parametrize("global_batching_method", global_batching_methods)
    def test_global_batching(self, dataset, device, global_batching_method):
        custom_args = [
            ("batching.global_batching.used_method", global_batching_method),
            ("batching.global_batching.global_batching_batch_size", 1000),
        ]
        bs = None
        if global_batching_method == "edges":
            bs = 1000
        elif global_batching_method == "minutes":
            bs = 10
        if bs:
            custom_args.append(("batching.global_batching.global_batching_batch_size", bs))

        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("intra_graph_batching_method", intra_graph_batching_methods)
    def test_intra_graph_batching(self, dataset, device, intra_graph_batching_method):
        custom_args = [
            (
                "batching.intra_graph_batching.used_methods",
                intra_graph_batching_method,
            ),
            (
                "batching.intra_graph_batching.edges.intra_graph_batch_size",
                200,
            ),
        ]
        if "tgn_last_neighbor" not in intra_graph_batching_method:
            custom_args.append(("training.encoder.used_methods", "graph_attention"))

        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("inter_graph_batching_method", inter_graph_batching_methods)
    def test_inter_graph_batching(self, dataset, device, inter_graph_batching_method):
        custom_args = [
            (
                "batching.inter_graph_batching.used_method",
                inter_graph_batching_method,
            ),
            ("batching.inter_graph_batching.inter_graph_batch_size", 2),
            ("batching.intra_graph_batching.used_methods", "none"),
            ("training.encoder.used_methods", "graph_attention"),
        ]

        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)


class TestSystems:
    systems = [
        "velox",
        "orthrus",
        "orthrus_non_snooped",
        "flash",
        "kairos",
        "magic",
        "nodlink",
        "threatrace",
        "rcaid",
        "ocrapt",
        "shadewatcher",
        "prographer",
    ]

    @pytest.mark.parametrize("system", systems)
    def test_systems(self, dataset, device, system):
        cfg = prepare_cfg(system, dataset, device=device)
        main.main(cfg)

    def test_shadewatcher_node_id_embedding(self, dataset, device):
        # ShadeWatcher with a learned embedding of each entity ID concatenated to its features
        custom_args = [("training.encoder.used_methods", "shadewatcher,node_id_embedding")]
        cfg = prepare_cfg("shadewatcher", dataset, device=device, custom_args=custom_args)
        main.main(cfg)


class TestEvaluationFigures:
    @pytest.mark.parametrize(
        "evaluation,figure",
        [
            ("node_evaluation", "adp_curve_model_epoch_0.png"),
            ("edge_evaluation", "adp_curve_model_epoch_0.png"),
            ("tw_evaluation", "pr_curve_model_epoch_0.png"),
            ("node_tw_evaluation", "pr_curve_model_epoch_0.png"),
        ],
    )
    def test_evaluation_figures(self, dataset, device, evaluation, figure):
        custom_args = [("evaluation.used_method", evaluation)]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args, save_figures=True)
        main.main(cfg)
        assert os.path.exists(os.path.join(cfg.evaluation._precision_recall_dir, figure))


class TestPretrainedEncoders:
    # Pretraining runs on the test dataset with a small budget, so no pretrained weights are needed.
    # The language models downloaded from HuggingFace (gpt2_pretrained, llama3_pretrained,
    # opt_pretrained) are not tested.
    model_types = ["spider", "bert", "deepwalk", "graphmae"]

    @staticmethod
    def pretraining_args(dataset, model_type="spider"):
        # Token budget of about two epochs on the test graphs (each epoch ends with a KNN
        # evaluation, which dominates the pretraining time). An epoch of spider has ~7k tokens and
        # its best checkpoints are only saved at the end of an epoch.
        pretrain_tokens = 10000 if model_type == "spider" else 1500
        return [
            ("featurization.pretrained.pretrain_datasets", dataset),
            ("featurization.pretrained.training.pretrain_tokens", pretrain_tokens),
            ("featurization.pretrained.training.warmup_tokens", pretrain_tokens // 10),
            ("featurization.pretrained.deepwalk.epochs", 1),
            ("featurization.pretrained.graphmae.epochs", 1),
        ]

    # Model types that do not use the default BPE tokenization
    tokenizations = {"deepwalk": "whole_label", "node2vec": "whole_label"}

    @pytest.mark.parametrize("model_type", model_types)
    def test_pretrained_featurization(self, dataset, device, model_type):
        custom_args = self.pretraining_args(dataset, model_type) + [
            ("featurization.pretrained.model_type", model_type),
            ("featurization.tokenization.used_method", self.tokenizations.get(model_type, "bpe")),
        ]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_continue_pretraining(self, dataset, device):
        custom_args = self.pretraining_args(dataset) + [
            ("feat_inference.continue_pretrain", True),
            ("feat_inference.continue_pretrain_epochs", 1),
        ]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_missing_weights_path(self, dataset, device):
        custom_args = [("featurization.pretrained.weights_path", "/nonexistent/weights")]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
        with pytest.raises(FileNotFoundError, match="weights_path not found"):
            main.main(cfg)

    def test_rename_attack(self, dataset, device):
        custom_args = self.pretraining_args(dataset) + [
            ("feat_inference.rename_attack.enabled", True),
            ("feat_inference.rename_attack.entities", "main, sendmail"),
            ("feat_inference.rename_attack.target_node_type", "subject"),
            ("feat_inference.rename_attack.top_k", 100),
        ]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_finetune_as_detector(self, dataset, device):
        # cybergfm pretrains a BERT
        custom_args = self.pretraining_args(dataset, "bert") + [("training.pretrained.finetune_epochs", 1)]
        cfg = prepare_cfg("cybergfm", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_supervised_fine_tuning(self, dataset, device):
        # The attack labels are embedded by feat_inference, which may be cached by a previous test
        custom_args = self.pretraining_args(dataset) + [("force_restart", "feat_inference")]
        cfg = prepare_cfg("pretrained_supervised", dataset, device=device, custom_args=custom_args)
        main.main(cfg)


class TestOptions:
    def test_stable_optim(self, dataset, device):
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=[("training.stable_optim", True)])
        main.main(cfg)

    def test_fuse_duplicate_edges_training(self, dataset, device):
        custom_args = [("training.fuse_duplicate_edges_training", True)]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("threshold_method", ["percentile", "fixed_zero"])
    def test_threshold_methods(self, dataset, device, threshold_method):
        custom_args = [("evaluation.node_evaluation.threshold_method", threshold_method)]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_best_ap_at_10(self, dataset, device):
        custom_args = [("evaluation.best_model_selection", "best_ap@10")]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_node_label_options(self, dataset, device):
        custom_args = [("construction.null_label_tokens", True)] + [
            (f"construction.node_label_features.{node_type}", "auto")
            for node_type in ("subject", "file", "netflow")
        ]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)


class TestCombinations:
    """Combinations of modules and options that are not the default of any shipped system."""

    @pytest.mark.parametrize(
        "objectives",
        [
            "predict_node_type,predict_edge_type",  # node-level objective first
            "predict_edge_type,predict_node_type",
            "reconstruct_node_features,predict_edge_type",
        ],
    )
    def test_node_and_edge_level_objectives(self, dataset, device, objectives):
        cfg = prepare_cfg("tests", dataset, device=device, objective=objectives)
        main.main(cfg)

    @pytest.mark.parametrize(
        "custom_args",
        [
            [("training.decoder.predict_edge_type.loss", "BCE")],
            [
                ("training.decoder.predict_edge_type.loss", "BCE"),
                ("training.decoder.predict_edge_type.balanced_loss", True),
            ],
            [("training.decoder.predict_edge_type.balanced_loss", True)],
            *[
                [
                    ("training.decoder.predict_edge_type.loss", "AMS"),
                    ("training.decoder.predict_edge_type.AMS.version", version),
                    ("training.decoder.predict_edge_type.AMS.margin", 0.2),
                    ("training.decoder.predict_edge_type.AMS.scale", 30),
                ]
                for version in [1, 2]
            ],
            [("training.encoder.tgn.use_residual_norm", True)],
            # BPR loss of ShadeWatcher with the negatives of other systems
            [
                ("training.decoder.used_methods", "predict_edge_contrastive"),
                ("training.decoder.predict_edge_contrastive.loss", "bpr"),
            ],
            [
                ("training.decoder.used_methods", "predict_edge_contrastive"),
                ("training.decoder.predict_edge_contrastive.loss", "bpr"),
                ("training.decoder.predict_edge_contrastive.neg_sampling", "uniform"),
            ],
            [("batching.edge_features", "time_encoding")],
            [("evaluation.used_method", "tw_evaluation")],
            [
                ("triage.used_method", "depimpact"),
                ("triage.depimpact.used_method", "component"),
                ("triage.depimpact.score_method", "degree"),
                ("triage.depimpact.workers", 2),
            ],
        ],
    )
    def test_options(self, dataset, device, custom_args):
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("use_residual_norm", [False, True])
    def test_tgn_memory_with_node_features(self, dataset, device, use_residual_norm):
        from pidsmaker.tasks.batching import get_preprocessed_graphs
        from pidsmaker.utils.data_utils import materialize_batch

        custom_args = [
            ("training.encoder.tgn.use_memory", True),
            ("training.encoder.tgn.use_node_feats_in_gnn", True),
            ("training.encoder.tgn.use_residual_norm", use_residual_norm),
            ("batching.intra_graph_batching.tgn_last_neighbor.fix_buggy_orthrus_TGN", True),
        ]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

        # Row i of the TGN node features is node `n_id_tgn[i]`, as the rows of the TGN memory
        train_data, val_data, test_data, _ = get_preprocessed_graphs(cfg)
        for graphs in train_data + val_data + test_data:
            for g in graphs:
                g = materialize_batch(g)  # features may be stored in a shared table
                src, dst = g.reindexed_edge_index_tgn
                assert g.x_tgn.shape[0] == g.x_from_tgn.shape[0] == g.n_id_tgn.shape[0]
                assert torch.equal(g.n_id_tgn[src], g.original_edge_index[0])
                assert torch.equal(g.n_id_tgn[dst], g.original_edge_index[1])
                assert torch.allclose(g.x_tgn[src], g.x_src)
                assert torch.allclose(g.x_tgn[dst], g.x_dst)

    def test_predict_node_type_with_mlp_decoder(self, dataset, device):
        from pidsmaker.factory import objective_factory

        # The threatrace scores are computed by the inference of the training task, which may be cached
        custom_args = [
            ("evaluation.node_evaluation.threshold_method", "threatrace"),
            ("force_restart", "training"),
        ]
        cfg = prepare_cfg(
            "tests", dataset, device=device, objective="predict_node_type", custom_args=custom_args
        )
        main.main(cfg)

        # The decoder predicts one logit per node type
        objective = objective_factory(cfg, in_dim=8, graph_reindexer=None, device="cpu")[0].objective
        h = torch.randn(5, cfg.training.node_out_dim)
        assert objective.decoder(h).shape == (5, cfg.dataset.num_node_types)

    def test_global_minutes_and_inter_graph_batching(self, dataset, device):
        custom_args = [
            ("batching.global_batching.used_method", "minutes"),
            ("batching.global_batching.global_batching_batch_size", 10),
            ("batching.inter_graph_batching.used_method", "graph_batching"),
            ("batching.inter_graph_batching.inter_graph_batch_size", 2),
            ("batching.intra_graph_batching.used_methods", "none"),
            ("training.encoder.used_methods", "graph_attention"),
        ]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_uncertainty_hyperparameters_cfg(self, dataset):
        custom_args = [("experiment", "run_n_times_hyperparameters")]
        prepare_cfg("tests", dataset, custom_args=custom_args)

    def test_uncertainty_hyperparameter_deltas(self, dataset):
        import copy

        from pidsmaker.experiments.uncertainty import update_cfg_for_uncertainty_exp

        custom_args = [("experiment", "run_n_times_hyperparameters")]
        cfg = prepare_cfg("tests", dataset, custom_args=custom_args)
        delta = cfg.experiment.uncertainty.hyperparameter.delta
        lr = cfg.training.lr
        offsets = []
        for i in range(5):
            new_cfg = update_cfg_for_uncertainty_exp("hyperparameter", i, 5, copy.deepcopy(cfg), "lr")
            offsets.append(new_cfg.training.lr / lr - 1)
        assert offsets == pytest.approx([-2 * delta, -delta, 0, delta, 2 * delta])

    @pytest.mark.parametrize(
        "model,custom_args",
        [
            # gat does not read `flow`, which flash does not set
            (
                "flash",
                [
                    ("training.encoder.used_methods", "gat"),
                    ("training.encoder.gat.activation", "relu"),
                    ("training.encoder.gat.num_heads", 2),
                    ("training.encoder.gat.concat", True),
                    ("training.encoder.gat.num_layers", 2),
                ],
            ),
            (
                "tests",
                [
                    ("training.encoder.tgn.use_memory", True),
                    ("batching.intra_graph_batching.tgn_last_neighbor.fix_buggy_orthrus_TGN", True),
                ],
            ),
        ],
    )
    def test_valid_combinations_cfg(self, dataset, model, custom_args):
        prepare_cfg(model, dataset, custom_args=custom_args)

    @pytest.mark.parametrize(
        "model,custom_args,match",
        [
            (
                "tests",
                [
                    ("batching.edge_features", "time_encoding"),
                    ("training.encoder.used_methods", "sage"),
                    ("batching.intra_graph_batching.used_methods", "edges"),
                ],
                "Edge feature `time_encoding` is only available",
            ),
            # no decoder set in default.yml
            ("tests", [("training.decoder.used_methods", "one_class")], "training.decoder.one_class.decoder"),
            (
                "tests",
                [
                    ("training.decoder.used_methods", "predict_node_type"),
                    ("training.decoder.predict_node_type.decoder", "magic_gat"),
                ],
                "Decoder `magic_gat` is a graph decoder",
            ),
            ("tests", [("evaluation.node_evaluation.threshold_method", "magic")], "Threshold method `magic`"),
            ("tests", [("training_loop.run_evaluation", "best_epoch")], "best_epoch"),
            ("tests", [("transformation.used_methods", "undirected, dag")], "`dag` transformation after `undirected`"),
            ("tests", [("triage.used_method", "depimpact")], "triage.depimpact.used_method"),
            (
                "tests",
                [
                    ("triage.used_method", "depimpact"),
                    ("triage.depimpact.used_method", "component"),
                    ("triage.depimpact.score_method", "degree"),
                    ("triage.depimpact.workers", 2),
                    ("evaluation.used_method", "edge_evaluation"),
                ],
                "Triage `depimpact` requires `evaluation.used_method`",
            ),
            # kairos.yml does not include default.yml, which sets the tw_evaluation threshold
            ("kairos", [("evaluation.used_method", "tw_evaluation")], "evaluation.tw_evaluation.threshold_method"),
            (
                "flash",
                [
                    ("evaluation.used_method", "queue_evaluation"),
                    ("evaluation.queue_evaluation.used_method", "kairos_idf_queue"),
                    ("evaluation.queue_evaluation.queue_threshold", 20),
                ],
                "Queue evaluation requires edge-level losses",
            ),
            # flash.yml does not include default.yml, which sets the gat args
            ("flash", [("training.encoder.used_methods", "gat")], "Missing arguments of the `gat` encoder"),
            ("ocrapt", [("evaluation.node_evaluation.threshold_method", "threatrace")], "requires the `predict_node_type` objective"),
            (
                "tests",
                [
                    ("batching.global_batching.used_method", "minutes"),
                    ("batching.global_batching.global_batching_batch_size", 100),
                ],
                "a step larger than the window",
            ),
            (
                "tests",
                [
                    ("training.encoder.used_methods", "hetero_graph_transformer,tgn"),
                    ("batching.edge_features", "edge_type_triplet"),
                ],
                "do not support the `edge_type_triplet`",
            ),
            (
                "tests",
                [
                    ("training.encoder.used_methods", "hetero_graph_transformer,tgn"),
                    ("transformation.used_methods", "undirected"),
                ],
                "do not support the `undirected` transformation",
            ),
            ("tests", [("training.encoder.used_methods", "tgn")], "exactly one encoder"),  # no GNN wrapped by TGN
            (
                "tests",
                [
                    ("training.encoder.used_methods", "sage,gat"),
                    ("batching.intra_graph_batching.used_methods", "edges"),
                ],
                "exactly one encoder",
            ),
            # Without the fix, node features are not aligned with the TGN memory
            ("tests", [("training.encoder.tgn.use_memory", True)], "fix_buggy_orthrus_TGN=True"),
            (
                "tests",
                [
                    ("training.encoder.tgn.use_memory", True),
                    ("training.encoder.tgn.use_node_feats_in_gnn", False),
                    ("training.encoder.tgn.use_residual_norm", True),
                ],
                "use_residual_norm",
            ),
        ],
    )
    def test_invalid_combinations_cfg(self, dataset, model, custom_args, match):
        with pytest.raises(ValueError, match=match):
            prepare_cfg(model, dataset, custom_args=custom_args)

    def test_invalid_decoder_none_size(self, dataset, device):
        # The input size of the decoder is only known when building the model
        custom_args = [
            ("training.decoder.used_methods", "reconstruct_node_features"),
            ("training.decoder.reconstruct_node_features.decoder", "none"),
        ]
        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        with pytest.raises(ValueError, match="Decoder `none` of objective `reconstruct_node_features`"):
            main.main(cfg)


class TestProGrapher:
    """ProGrapher: graph2vec snapshot embeddings, TextRCNN and key indicators."""

    @pytest.mark.parametrize(
        "custom_args",
        [
            # Entity-level evaluation from the key indicators, with the top-4 RSGs of the paper
            [
                ("evaluation.used_method", "node_evaluation"),
                ("training.decoder.predict_snapshot_embedding.top_k", 4),
            ],
            # ReLU recurrence of the TextRCNN equations, transductive graph2vec
            [
                ("training.encoder.textrcnn.rnn_type", "rnn"),
                ("featurization.training_split", "all"),
                ("evaluation.used_method", "node_tw_evaluation"),
            ],
        ],
    )
    def test_prographer_options(self, dataset, device, custom_args):
        custom_args = custom_args + [("training.num_epochs", 2), ("featurization.epochs", 2)]
        cfg = prepare_cfg("prographer", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_textrcnn_requires_graph2vec(self, dataset, device):
        custom_args = [
            ("training.decoder.used_methods", "predict_snapshot_embedding"),
            ("batching.intra_graph_batching.used_methods", "none"),
        ]
        cfg = prepare_cfg(
            "tests", dataset, device=device, encoder="textrcnn", custom_args=custom_args
        )
        with pytest.raises(ValueError, match="graph2vec"):
            main.main(cfg)

    def test_rooted_subgraphs(self):
        import numpy as np

        from pidsmaker.featurization.featurization_methods.featurization_graph2vec import (
            rooted_subgraphs,
        )

        # p0 -READ-> f1, p0 -WRITE-> f2, and the same graph with other node and edge orders
        labels = rooted_subgraphs(
            ["subject", "file", "file"], np.array([0, 0]), np.array([1, 2]), ["READ", "WRITE"], 2
        )
        permuted = rooted_subgraphs(
            ["file", "file", "subject"], np.array([2, 2]), np.array([1, 0]), ["WRITE", "READ"], 2
        )
        assert labels.shape == (3, 3)
        assert list(labels[0]) == ["subject", "file", "file"]
        # Files differ by their edge types from depth 1, the process is the same in both graphs
        assert labels[1, 1] != labels[1, 2]
        assert list(permuted[:, 2]) == list(labels[:, 0])
        assert sorted(permuted[2, :2]) == sorted(labels[2, 1:])

    def test_textrcnn_short_sequences(self):
        from types import SimpleNamespace

        from pidsmaker.encoders import TextRCNN
        from pidsmaker.objectives import SnapshotEmbeddingPrediction

        torch.manual_seed(0)
        encoder = TextRCNN(emb_dim=8, hid_dim=4, num_layers=2, sequence_length=3, dropout=0.2)
        # RSG 0 is aligned with the deviation, RSG 1 opposed to it
        objective = SnapshotEmbeddingPrediction(torch.tensor([[1.0] * 8, [-1.0] * 8]))
        for i in range(5):  # fewer, then more snapshots than `sequence_length`
            batch = SimpleNamespace(
                graph_emb=torch.full((1, 8), float(i)),
                original_n_id=torch.tensor([3, 7]),
                # node 3: RSGs 1 (depth 0) and 0 (depth 1), node 7: RSG 1 and an unseen RSG
                snapshot_rsg=torch.tensor([[[3, 1, 0], [7, 1, -1]]], dtype=torch.int32),
            )
            res = encoder(batch=batch, edge_index=torch.tensor([[0], [1]]))
            assert res["h"].shape == (1, 8) and res["h_src"].shape == (1, 8)
            assert len(encoder.history) == min(i + 1, 3)
            loss = objective(h=res["h"], inference=False, batch=batch)["loss"]
            loss.backward()

        with torch.no_grad():
            pred = torch.full((1, 8), -10.0)  # the actual embedding (4) is above the prediction
            scores = objective(h=pred, inference=True, validation=True, batch=batch)["loss"]
        error = torch.linalg.vector_norm(batch.graph_emb - pred)
        # Node 3 is rooted at the top RSG: it gets the snapshot error, node 7 half of it
        assert torch.allclose(scores, torch.stack([error, error / 2]))
        assert objective.get_val_score() == pytest.approx(-error.item())
        encoder.reset_state()
        assert encoder.history == []
