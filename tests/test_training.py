import os
import sys
import pytest
import pathlib

import torch
import wandb
from torch_geometric.data import TemporalData

parent = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(os.path.join(parent, "config"))
sys.path.append(os.path.join(parent, "src"))

from config import *
import benchmark

def mock_input_graph(num_nodes=10_000, num_edges=1000):
    return TemporalData(
        src=torch.randint(low=0, high=num_nodes, size=(num_edges, )),
        dst=torch.randint(low=0, high=num_nodes, size=(num_edges, )),
        msg=torch.cat([
            torch.tensor([0, 0, 1]),
            torch.rand(size=(128, )),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.rand(size=(128, )),
        ]).repeat(num_edges, 1),
        t=torch.arange(num_edges),
    )
    
def store_data_to_disk(data, cfg):
    # Store the mock on disk to be loaded in training
    edge_embeds_dir = os.path.join(cfg.featurization.embed_edges._edge_embeds_dir, "train")
    os.makedirs(edge_embeds_dir, exist_ok=True)
    torch.save(data, os.path.join(edge_embeds_dir, "data.pkl"))
    
def prepare_cfg(model):
    input_args = [model, "THEIA_E3"]  # here we only care about the model, not the dataset as we'll use a mock data
    args, unknown_args = get_runtime_required_args(return_unknown_args=True, args=input_args)
    
    # We force the creation of a new path only for tests by setting a different value to an arg
    args.__dict__["detection.gnn_training.num_epochs"] = 1
    args.__dict__["featurization.embed_edges.include_edge_type"] = "mock"
    cfg = get_yml_cfg(args)
    
    # We bypass all other steps and run only the training part
    cfg._subtasks_should_restart_with_deps = \
        [(k, k == 'gnn_training') for k, v in cfg._subtasks_should_restart_with_deps]
        
    return cfg

@pytest.mark.parametrize(
    "model",
    [
        "kairos",
        "kairos_plus_plus",
        "threatrace",
        "playground",
    ]
)
def test_training(model):
    # Avoids error from wandb.log calls
    wandb.init(mode="disabled")
    
    cfg = prepare_cfg(model)
    data = mock_input_graph()
    store_data_to_disk(data, cfg)
    
    benchmark.main(cfg, save_model=False)
