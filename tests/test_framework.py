import os
import shutil
from itertools import product

import pytest
import wandb

import pidsmaker.config as config
from pidsmaker import benchmark
from pidsmaker.config import (
    get_runtime_required_args,
    get_yml_cfg,
)


def prepare_cfg(
    model,
    dataset,
    transformation=None,
    featurization=None,
    encoder=None,
    objective=None,
    decoder=None,
):
    input_args = [model, dataset]
    args, _ = get_runtime_required_args(return_unknown_args=True, args=input_args)

    if transformation:
        args.__dict__["preprocessing.transformation.used_methods"] = transformation
    if featurization:
        args.__dict__["featurization.embed_nodes.used_method"] = featurization
    if encoder:
        args.__dict__["detection.gnn_training.encoder.used_methods"] = encoder
    if objective:
        args.__dict__["detection.gnn_training.decoder.used_methods"] = objective
    if decoder and objective:
        args.__dict__[f"detection.gnn_training.decoder.{objective}.decoder"] = decoder

    if encoder and "tgn" not in encoder:
        args.__dict__["detection.graph_preprocessing.intra_graph_batching.used_methods"] = "edges"

    cfg = get_yml_cfg(args)
    cfg._test_mode = True

    return cfg


@pytest.fixture(scope="session", autouse=True)
def framework_setup_teardown():
    # Runs before tests
    config.ROOT_ARTIFACT_DIR = os.path.join(config.ROOT_ARTIFACT_DIR, "tests/")
    shutil.rmtree(config.ROOT_ARTIFACT_DIR, ignore_errors=True)
    wandb.init(mode="disabled")

    yield
    # Runs after tests
    shutil.rmtree(config.ROOT_ARTIFACT_DIR)


@pytest.fixture(scope="class")
def dataset():
    return "CLEARSCOPE_E3"


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
                benchmark.main(cfg)
        else:
            cfg = prepare_cfg("tests", dataset, transformation=transformation)
            benchmark.main(cfg)

    @pytest.mark.parametrize("transformation", transformations)
    def test_transformations(self, dataset, transformation):
        cfg = prepare_cfg("nodlink", dataset, transformation=transformation)
        benchmark.main(cfg)


class TestFeaturization:
    featurizations = [
        "feature_word2vec",
        "doc2vec",
        "hierarchical_hashing",
        "only_type",
        "only_ones",
        "fasttext",
        "flash",
        "word2vec",
        "temporal_rw",
    ]

    @pytest.mark.parametrize("featurization", featurizations)
    def test_featurizations(self, dataset, featurization):
        cfg = prepare_cfg("tests", dataset, featurization=featurization)
        benchmark.main(cfg)


class TestEncoderObjective:
    encoders = [
        "graph_attention",
        "sage",
        "rcaid_gat",
        "magic_gat",
        "sum_aggregation",
        "custom_mlp",
        "none",
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
    def test_encoder_objective_pairs(self, dataset, encoder, objective):
        cfg = prepare_cfg("tests", dataset, encoder=encoder, objective=objective)
        benchmark.main(cfg)

    @pytest.mark.parametrize("encoder,objective", list(product(encoders, objectives)))
    def test_encoder_tgn_objective_pairs(self, dataset, encoder, objective):
        encoder_combined = f"{encoder},tgn"
        cfg = prepare_cfg("tests", dataset, encoder=encoder_combined, objective=objective)
        benchmark.main(cfg)


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
    def test_decoder_objective_pairs_node_level_success(self, dataset, decoder, objective):
        cfg = prepare_cfg("tests", dataset, decoder=decoder, objective=objective)
        benchmark.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(edge_decoders, edge_level_objectives))
    )
    def test_decoder_objective_pairs_edge_level_success(self, dataset, decoder, objective):
        cfg = prepare_cfg("tests", dataset, decoder=decoder, objective=objective)
        benchmark.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(node_decoders, edge_level_objectives))
    )
    def test_decoder_objective_pairs_node_level_fail(self, dataset, decoder, objective):
        with pytest.raises(ValueError):
            cfg = prepare_cfg("tests", dataset, decoder=decoder, objective=objective)
            benchmark.main(cfg)

    @pytest.mark.parametrize(
        "decoder,objective", list(product(edge_decoders, node_level_objectives))
    )
    def test_decoder_objective_pairs_edge_level_fail(self, dataset, decoder, objective):
        with pytest.raises(ValueError):
            cfg = prepare_cfg("tests", dataset, decoder=decoder, objective=objective)
            benchmark.main(cfg)
