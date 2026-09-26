import os
import shutil
from itertools import product

import pytest
import wandb

from pidsmaker import main
from pidsmaker.config import (
    ENCODERS_CFG,
    get_runtime_required_args,
    get_yml_cfg,
)

TESTS_ARTIFACT_DIR = os.path.join("/home/artifacts/", "tests2/")


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

    if custom_args:
        for k, v in custom_args:
            args.__dict__[k] = v

    cfg = get_yml_cfg(args)
    cfg._test_mode = True

    if device == "cpu":
        cfg._use_cpu = True

    return cfg


@pytest.fixture(scope="session", autouse=True)
def framework_setup_teardown():
    # Runs before tests
    shutil.rmtree(TESTS_ARTIFACT_DIR, ignore_errors=True)
    wandb.init(mode="disabled")

    yield
    # Runs after tests
    shutil.rmtree(TESTS_ARTIFACT_DIR)


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
    ]

    @pytest.mark.parametrize("featurization", featurizations)
    def test_featurizations(self, dataset, featurization):
        cfg = prepare_cfg("tests", dataset, featurization=featurization)
        main.main(cfg)


class TestEncoderObjective:
    # `tgn` is a wrapper, covered by test_encoder_tgn_objective_pairs below.
    encoders = [e for e in ENCODERS_CFG.keys() if e != "tgn"]
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
    ]

    @pytest.mark.parametrize("system", systems)
    def test_systems(self, dataset, device, system):
        cfg = prepare_cfg(system, dataset, device=device)
        main.main(cfg)


class TestPretrainedEncoders:
    # Pretraining runs on the test dataset with a small budget, so no pretrained weights are needed.
    # The language models downloaded from HuggingFace (gpt2_pretrained, llama3_pretrained,
    # opt_pretrained) are not tested.
    model_types = ["spider", "bert", "deepwalk", "graphmae"]

    @staticmethod
    def pretraining_args(dataset):
        return [
            ("featurization.pretrained.pretrain_datasets", dataset),
            ("featurization.pretrained.training.pretrain_tokens", 20000),
            ("featurization.pretrained.training.warmup_tokens", 2000),
            ("featurization.pretrained.deepwalk.epochs", 1),
            ("featurization.pretrained.graphmae.epochs", 1),
        ]

    @pytest.mark.parametrize("model_type", model_types)
    def test_pretrained_featurization(self, dataset, device, model_type):
        custom_args = self.pretraining_args(dataset) + [("featurization.pretrained.model_type", model_type)]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    def test_continue_pretraining(self, dataset, device):
        custom_args = self.pretraining_args(dataset) + [
            ("feat_inference.continue_pretrain", True),
            ("feat_inference.continue_pretrain_epochs", 1),
        ]
        cfg = prepare_cfg("pretrained_velox", dataset, device=device, custom_args=custom_args)
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
        custom_args = self.pretraining_args(dataset) + [("training.pretrained.finetune_epochs", 1)]
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
