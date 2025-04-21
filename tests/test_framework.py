import os
import sys
import pathlib
import pytest
import shutil
from itertools import product

import wandb

parent = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(os.path.join(parent, "src"))

import benchmark
import config
from config import get_runtime_required_args, get_yml_cfg


def prepare_cfg(model, dataset, featurization=None, encoder=None, decoder=None):
    input_args = [model, dataset]
    args, _ = get_runtime_required_args(return_unknown_args=True, args=input_args)
    
    if featurization:
        args.__dict__["featurization.embed_nodes.used_method"] = featurization
    if encoder:
        args.__dict__["detection.gnn_training.encoder.used_methods"] = encoder
    if decoder:
        args.__dict__["detection.gnn_training.decoder.used_methods"] = decoder
        
    if encoder and "tgn" not in encoder:
        args.__dict__["detection.graph_preprocessing.intra_graph_batching.used_methods"] = "edges"
    
    cfg = get_yml_cfg(args)
    cfg._test_mode = True
    
    return cfg

@pytest.fixture(scope="session", autouse=True)
def framework_setup_teardown():
    # Runs before tests
    config.ROOT_ARTIFACT_DIR = os.path.join(config.ROOT_ARTIFACT_DIR, "tests/")
    wandb.init(mode="disabled")

    yield
    # Runs after tests
    shutil.rmtree(config.ROOT_ARTIFACT_DIR)

@pytest.fixture(scope="class")
def dataset():
    return "CLEARSCOPE_E3"


encoders = [
    "graph_attention",
    "sage",
    "rcaid_gat",
    "magic_gat",
    "sum_aggregation",
    "custom_mlp",
    "none",
]
decoders = [
    "predict_node_type",
    "reconstruct_node_embeddings",
    "reconstruct_node_features",
    "predict_edge_type",
    "reconstruct_edge_embeddings",
]
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

class TestFramework:
    """Test suite for the framework."""

    @pytest.mark.parametrize("featurization", featurizations)
    def test_featurizations(self, dataset, featurization):
        cfg = prepare_cfg("tests", dataset, featurization=featurization)
        benchmark.main(cfg)

    @pytest.mark.parametrize("encoder,decoder", list(product(encoders, decoders)))
    def test_encoder_decoder_pairs(self, dataset, encoder, decoder):
        cfg = prepare_cfg("tests", dataset, encoder=encoder, decoder=decoder)
        benchmark.main(cfg)

    @pytest.mark.parametrize("encoder,decoder", list(product(encoders, decoders)))
    def test_encoder_tgn_decoder_pairs(self, dataset, encoder, decoder):
        encoder_combined = f"{encoder},tgn"
        cfg = prepare_cfg("tests", dataset, encoder=encoder_combined, decoder=decoder)
        benchmark.main(cfg)
        