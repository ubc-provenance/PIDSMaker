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

TESTS_ARTIFACT_DIR = os.path.join("/home/artifacts/", "tests/")


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
    args.__dict__["artifact_dir_in_container"] = TESTS_ARTIFACT_DIR

    if transformation:
        args.__dict__["preprocessing.transformation.used_methods"] = transformation
    if featurization:
        args.__dict__["featurization.feat_training.used_method"] = featurization
    if encoder:
        args.__dict__["detection.gnn_training.encoder.used_methods"] = encoder
    if objective:
        args.__dict__["detection.gnn_training.decoder.used_methods"] = objective
    if decoder and objective:
        args.__dict__[f"detection.gnn_training.decoder.{objective}.decoder"] = decoder

    if encoder and "tgn" not in encoder:
        args.__dict__["detection.graph_preprocessing.intra_graph_batching.used_methods"] = "edges"

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
        cfg = prepare_cfg(
            "tests", dataset, device=device, encoder=encoder_combined, objective=objective
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
            ("detection.graph_preprocessing.global_batching.used_method", global_batching_method),
            ("detection.graph_preprocessing.global_batching.global_batching_batch_size", 1000),
        ]
        bs = None
        if global_batching_method == "edges":
            bs = 1000
        elif global_batching_method == "minutes":
            bs = 10
        if bs:
            custom_args.append(
                ("detection.graph_preprocessing.global_batching.global_batching_batch_size", bs)
            )

        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("intra_graph_batching_method", intra_graph_batching_methods)
    def test_intra_graph_batching(self, dataset, device, intra_graph_batching_method):
        custom_args = [
            (
                "detection.graph_preprocessing.intra_graph_batching.used_methods",
                intra_graph_batching_method,
            ),
            (
                "detection.graph_preprocessing.intra_graph_batching.edges.intra_graph_batch_size",
                200,
            ),
        ]
        if "tgn_last_neighbor" not in intra_graph_batching_method:
            custom_args.append(("detection.gnn_training.encoder.used_methods", "graph_attention"))

        cfg = prepare_cfg("tests", dataset, device=device, custom_args=custom_args)
        main.main(cfg)

    @pytest.mark.parametrize("inter_graph_batching_method", inter_graph_batching_methods)
    def test_inter_graph_batching(self, dataset, device, inter_graph_batching_method):
        custom_args = [
            (
                "detection.graph_preprocessing.inter_graph_batching.used_method",
                inter_graph_batching_method,
            ),
            ("detection.graph_preprocessing.inter_graph_batching.inter_graph_batch_size", 2),
            ("detection.graph_preprocessing.intra_graph_batching.used_methods", "none"),
            ("detection.gnn_training.encoder.used_methods", "graph_attention"),
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
    ]

    @pytest.mark.parametrize("system", systems)
    def test_systems(self, dataset, device, system):
        cfg = prepare_cfg(system, dataset, device=device)
        main.main(cfg)
