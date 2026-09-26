"""SPIDER model architectures."""

from .bert import (
    ProvenanceBERT, ProvenanceBERTFineTuneCLS, ProvenanceBERTFineTuneLP,
    FineTuneCLS, FineTuneLP,
    get_bert_config, compute_cls_loss, MODEL_SIZES,
)
from .roberta import ProvenanceRoBERTa, ProvenanceRoBERTaFineTuneCLS, ProvenanceRoBERTaFineTuneLP, get_roberta_config
from .modernbert import ProvenanceModernBERT, ProvenanceModernBERTFineTuneCLS, ProvenanceModernBERTFineTuneLP, get_modernbert_config
from .ropebert import ProvenanceRoPEBERT, ProvenanceRoPEBERTFineTuneCLS, ProvenanceRoPEBERTFineTuneLP, get_ropebert_config
from .llama import ProvenanceLLaMA, get_llama_config
from .t5 import ProvenanceT5, get_t5_config
from .behavior import ProvenanceBehaviorModel, get_behavior_encoder_config, behavior_combined_loss
from .gnn_distill import ProvenanceGNNDistill, NeighborhoodGNNTeacher, get_gnn_distill_encoder_config, build_gnn_distill_batch, build_edge_type_encoder
from .spider import ProvenanceGNNCluster, GNNClusterTeacher, get_spider_encoder_config, supervised_contrastive_loss, StudentSignatureHead, StudentClassHead, TeacherClassHead
from .graphmae import GraphMAE, sce_loss, T5NodeEncoder, neighborhoods_to_pyg_batch, save_graphmae, load_graphmae
from .gae import GAE, save_gae, load_gae
from .dgi import DGI, save_dgi, load_dgi
from .tgn import ProvenanceTGN, EdgeTypeClassifier
from .deepwalk import walks_to_label_sentences, train_deepwalk, save_deepwalk, load_deepwalk, deepwalk_embeddings
from .hf_pretrained import HF_MODEL_TYPES, get_hf_model_name, get_hf_hidden_size, pretokenize_walks_hf
