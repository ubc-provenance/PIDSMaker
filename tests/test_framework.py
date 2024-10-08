import argparse
import os
import sys
import pathlib
import shutil
import wandb

parent = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(os.path.join(parent, "config"))
sys.path.append(os.path.join(parent, "src"))

import config
from config import *
import benchmark


# === Utils ===
def prepare_cfg(model, dataset, featurization_method, encoder, decoder):
    input_args = [model, dataset]
    args, unknown_args = get_runtime_required_args(return_unknown_args=True, args=input_args)
    
    args.__dict__["featurization.embed_nodes.used_method"] = featurization_method
    args.__dict__["detection.gnn_training.encoder.used_methods"] = encoder or "tgn,graph_attention"
    args.__dict__["detection.gnn_training.decoder.used_methods"] = decoder or "predict_edge_type"
    
    cfg = get_yml_cfg(args)
    cfg._test_mode = True
    
    return cfg

def run_encoders(model, dataset, featurization_method):
    # Test each encoder
    encoders = ["graph_attention", "sage"]
    for method in encoders:
        # NOTE: cfg needs to be created before each call to benchmark to be updated
        cfg = prepare_cfg(model, dataset, featurization_method, encoder=method, decoder=None)
        benchmark.main(cfg)
        
    # Test each encoder with TGN
    for method in encoders:
        cfg = prepare_cfg(model, dataset, featurization_method, encoder=",".join([method, "tgn"]), decoder=None)
        benchmark.main(cfg)

def run_decoders(model, dataset, featurization_method):
    # Test each decoder
    decoders = ["reconstruct_node_features", "reconstruct_node_embeddings", "reconstruct_edge_embeddings", "predict_edge_type"]
    for method in decoders:
        try:
            cfg = prepare_cfg(model, dataset, featurization_method, encoder=None, decoder=method)
            benchmark.main(cfg)
        except EnvironmentError: # ignore edge cases errors
            pass


# === Tests ===
def test_yml_files(dataset: str):
    """
    Tests that YML file configurations do not generate errors.
    """
    print(f"Testing YML files...")
    # Avoids error from wandb.log calls
    wandb.init(mode="disabled")
    
    models = [
        "kairos",
        "kairos_plus_plus",
        "threatrace",
    ]
    
    for model in models:
        print(f"Testing {model}...")
        cfg = prepare_cfg(model=model, dataset=dataset, encoder=None, decoder=None)
        benchmark.main(cfg)


def test_whole_framework(dataset: str):
    """
    Tests all subtasks of the framework with different featurization methods.
    """
    print(f"Testing whole framework...")
    # Avoids error from wandb.log calls
    wandb.init(mode="disabled")
    
    featurization_methods = [
        "temporal_rw", "word2vec", "doc2vec", "feature_word2vec", "hierarchical_hashing", "only_type", "magic", "flash",
    ]
    # "word2vec"
    base_model = "tests"
    
    for featurization_method in featurization_methods:
        run_encoders(base_model, dataset, featurization_method)
        run_decoders(base_model, dataset, featurization_method)
    

if __name__ == "__main__":
    config.ROOT_ARTIFACT_DIR = os.path.join(config.ROOT_ARTIFACT_DIR, "tests/")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="Name of the dataset to run the tests on.")
    parser.add_argument('--restart', action="store_true", help="Removes all existing files and start tests from beginning.")
    args, _ = parser.parse_known_args()

    if args.restart:
        shutil.rmtree(config.ROOT_ARTIFACT_DIR, ignore_errors=True)
    
    test_whole_framework(args.dataset)
    # test_yml_files(args.dataset)
    
    shutil.rmtree(config.ROOT_ARTIFACT_DIR)
