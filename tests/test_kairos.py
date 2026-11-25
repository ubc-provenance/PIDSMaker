"""Tests of the parts of Kairos that follow the original implementation, without a database."""

import pytest
import torch
from yacs.config import CfgNode as CN

from pidsmaker.encoders.custom_mlp import build_mlp_from_string
from pidsmaker.encoders.graph_attention import GraphAttentionEmbedding
from pidsmaker.tgn import LastNeighborLoader
from pidsmaker.utils.data_utils import _msg_has_edge_type

pytestmark = pytest.mark.no_database


def decoder_cfg(objective, edge_type_in_msg=None):
    cfg = CN()
    cfg.training = CN()
    cfg.training.decoder = CN()
    cfg.training.decoder.used_methods = objective
    cfg.training.decoder.predict_edge_type = CN()
    cfg.training.decoder.predict_edge_type.edge_type_in_msg = edge_type_in_msg
    return cfg


def test_msg_has_edge_type():
    # Removed from the messages when predicting edge types, unless kept as in Kairos
    assert not _msg_has_edge_type(decoder_cfg("predict_edge_type"))
    assert not _msg_has_edge_type(decoder_cfg("predict_edge_type", False))
    assert _msg_has_edge_type(decoder_cfg("predict_edge_type", True))
    assert _msg_has_edge_type(decoder_cfg("predict_edge_contrastive"))


def encoder_cfg(encoders, objectives, use_memory=True, use_node_feats_in_gnn=False):
    cfg = decoder_cfg(objectives)
    cfg.training.decoder.use_few_shot = False
    cfg.training.encoder = CN()
    cfg.training.encoder.used_methods = encoders
    cfg.training.encoder.tgn = CN()
    cfg.training.encoder.tgn.use_memory = use_memory
    cfg.training.encoder.tgn.use_node_feats_in_gnn = use_node_feats_in_gnn
    return cfg
