"""
Dynamic BERT embedding generator for GNN training with temporal correctness.

For each edge event at timestamp t, samples backward walks ending at src/dst
nodes using only edges before timestamp t, then encodes through frozen BERT.
"""

import os
import torch
import numpy as np
from collections import defaultdict

from .models.bert import ProvenanceBERT, get_bert_config
from .models.modernbert import ProvenanceModernBERT, get_modernbert_config
from .models.ropebert import ProvenanceRoPEBERT, get_ropebert_config
from .data.sampler import ProvenanceWalkSampler
from .data.tokenizer_bpe import ProvenanceTokenizerBPE as ProvenanceTokenizer


class SpiderEmbeddingGenerator:
    """Generates node embeddings on-the-fly using pretrained frozen BERT.

    Ensures temporal correctness by sampling backward walks based on edge timestamps.
    """

    def __init__(self, cfg, device=None):
        """Load pretrained BERT model and tokenizer.

        Args:
            cfg: Configuration object
            device: torch device (defaults to cuda if available)
        """
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model and tokenizer
        self.model, self.tokenizer = self._load_pretrained_bert()
        self.model.eval()  # Frozen for inference

        # SPIDER walks config
        spider_cfg = cfg.featurization.pretrained
        walks_cfg = spider_cfg.walks
        self.walk_length = walks_cfg.walk_length
        self.num_walks = walks_cfg.num_walks
        self.time_weight = walks_cfg.time_weight
        self.half_life = walks_cfg.half_life
        self.diversity_weight = walks_cfg.diversity_weight

    def _load_pretrained_bert(self):
        """Load the pretrained BERT model and tokenizer."""
        model_dir = self.cfg.featurization._model_dir
        spider_cfg = self.cfg.featurization.pretrained

        model_type = spider_cfg.model_type
        model_size = spider_cfg.model_size
        max_seq_len = spider_cfg.tokenizer.max_seq_len

        model_path = os.path.join(model_dir, f"pretrain_{model_size}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pretrained BERT model not found at {model_dir}. "
                "Run featurization training first."
            )

        # Load tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer.pt")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        tokenizer = ProvenanceTokenizer(self.cfg)
        tokenizer.load(tokenizer_path)

        # Create and load model based on model_type
        if model_type == "ropebert":
            rope_theta = spider_cfg.mlm.ropebert.rope_theta
            bert_config = get_ropebert_config(
                vocab_size=tokenizer.vocab_size,
                model_size=model_size,
                max_position_embeddings=max_seq_len,
                rope_theta=rope_theta,
            )
            model = ProvenanceRoPEBERT(bert_config)
        elif model_type == "modernbert":
            mb_cfg = spider_cfg.mlm.modernbert
            bert_config = get_modernbert_config(
                vocab_size=tokenizer.vocab_size,
                model_size=model_size,
                max_position_embeddings=max_seq_len,
                global_attn_every_n_layers=mb_cfg.global_attn_every_n_layers,
                local_attention_window=mb_cfg.local_attention_window,
            )
            model = ProvenanceModernBERT(bert_config)
        elif model_type == "gnn_distill":
            from .models.gnn_distill import ProvenanceGNNDistill, get_gnn_distill_encoder_config
            distill_cfg = spider_cfg.gnn_distill
            enc_config = get_gnn_distill_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
            model = ProvenanceGNNDistill(enc_config, distill_cfg.hidden_dim)
        elif model_type == "spider":
            # Student is identical to gnn_distill (T5 + DistillProjection)
            from .models.spider import ProvenanceGNNCluster, get_spider_encoder_config
            gc_cfg = spider_cfg.spider
            enc_config = get_spider_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
            model = ProvenanceGNNCluster(enc_config, gc_cfg.hidden_dim)
        elif model_type == "behavior_cluster":
            from .models.behavior import ProvenanceBehaviorModel, get_behavior_encoder_config
            bc_cfg = spider_cfg.behavior_cluster
            # We need the vocab size for num_labels, but at inference only the encoder matters.
            # Load the behavior vocab to get the label count.
            from .data.behavior_signatures import BehaviorLabelVocab
            bvocab_path = os.path.join(model_dir, "behavior_vocab.txt")
            bvocab = BehaviorLabelVocab()
            if os.path.exists(bvocab_path):
                bvocab.load(bvocab_path)
            num_labels = max(bvocab.size, 1)
            enc_config = get_behavior_encoder_config(tokenizer.vocab_size, model_size, max_seq_len)
            model = ProvenanceBehaviorModel(enc_config, num_labels=num_labels, proj_dim=bc_cfg.proj_dim)
        else:  # bert (traditional)
            bert_config = get_bert_config(
                vocab_size=tokenizer.vocab_size,
                model_size=model_size,
                max_position_embeddings=max_seq_len,
            )
            model = ProvenanceBERT(bert_config)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model, tokenizer

    def get_batch_embeddings(self, nodes, graph, indexid2msg, timestamps=None):
        """Generate embeddings for a batch of nodes at specific timestamps.

        For temporal correctness, only uses edges before each node's timestamp.

        Args:
            nodes: List of node IDs (can contain duplicates with different timestamps)
            graph: NetworkX graph containing these nodes
            indexid2msg: Mapping from node ID to (node_type, label_str)
            timestamps: Optional list of timestamps (one per node). If None, uses all edges.

        Returns:
            List of embeddings (numpy arrays) in same order as input nodes
        """

        # Create sampler for this graph
        sampler = ProvenanceWalkSampler(
            graph,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            time_weight=self.time_weight,
            half_life=self.half_life,
            diversity_weight=self.diversity_weight,
        )

        # Collect all walks with their node indices (not IDs, to handle duplicates)
        all_walks = []  # List of (node_index, walk_nodes, walk_edge_types)

        for i, node in enumerate(nodes):
            timestamp = timestamps[i] if timestamps is not None else None

            # Sample backward walks ending at this node (only past edges)
            for _ in range(self.num_walks):
                walk_nodes, walk_edge_types = sampler.sample_context_walk(
                    target_node=node,
                    walk_length=self.walk_length,
                    max_ts=timestamp,  # Only use edges before this time
                )
                if len(walk_nodes) > 1:  # Skip single-node walks
                    all_walks.append((i, walk_nodes, walk_edge_types))  # Store index, not node ID

        if not all_walks:
            # No valid walks, return zero vectors as list
            return [np.zeros(self.model.config.hidden_size) for _ in nodes]

        # Tokenize all walks
        tokenized_data = []  # List of (node_index, token_ids)
        for node_idx, walk_nodes, walk_edge_types in all_walks:
            token_ids, _, _ = self.tokenizer.tokenize_walk(walk_nodes, walk_edge_types, indexid2msg)
            if token_ids:
                tokenized_data.append((node_idx, token_ids))

        if not tokenized_data:
            # Tokenization failed, return zero vectors as list
            return [np.zeros(self.model.config.hidden_size) for _ in nodes]

        # Batch process through BERT
        batch_size = self.cfg.featurization.pretrained.training.batch_size
        all_cls_embeddings = []  # List of (node_index, cls_embedding)

        for batch_start in range(0, len(tokenized_data), batch_size):
            batch_data = tokenized_data[batch_start : batch_start + batch_size]

            # Pad to max length in batch
            max_len = min(max(len(token_ids) for _, token_ids in batch_data), self.tokenizer.max_seq_len)
            curr_batch_size = len(batch_data)

            input_ids = torch.full((curr_batch_size, max_len), self.tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros(curr_batch_size, max_len, dtype=torch.bool)

            for i, (node_idx, token_ids) in enumerate(batch_data):
                seq_len = min(len(token_ids), max_len)
                input_ids[i, :seq_len] = torch.tensor(token_ids[:seq_len])
                attention_mask[i, :seq_len] = True

            # Run through frozen BERT
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                hidden_states = self.model.modified_fwd(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=torch.full_like(input_ids, -100),  # Dummy labels (not used with skip_cls)
                    skip_cls=True,
                )

            # Extract [CLS] embeddings
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # [batch_size, hidden_size]

            for i, (node_idx, _) in enumerate(batch_data):
                all_cls_embeddings.append((node_idx, cls_embeddings[i]))

        # Average embeddings per node INDEX (handles same node at different timestamps correctly)
        node_index_embeddings = defaultdict(list)
        for node_idx, cls_emb in all_cls_embeddings:
            node_index_embeddings[node_idx].append(cls_emb)

        # Create final embeddings as list in same order as input nodes
        final_embeddings = []
        for node_idx in range(len(nodes)):
            if node_idx in node_index_embeddings:
                # Average across walks for this specific (node, timestamp) pair
                embeddings = node_index_embeddings[node_idx]
                node_embedding = np.mean(embeddings, axis=0)

                # Normalize
                norm = np.linalg.norm(node_embedding)
                if norm > 1e-12:
                    node_embedding = node_embedding / norm
            else:
                # No valid walks for this (node, timestamp) pair
                node_embedding = np.zeros(self.model.config.hidden_size)

            final_embeddings.append(node_embedding)

        return final_embeddings

    @property
    def embedding_dim(self):
        """Return the dimensionality of generated embeddings."""
        return self.model.config.hidden_size
