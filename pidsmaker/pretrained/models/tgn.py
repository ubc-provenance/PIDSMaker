"""
TGN (Temporal Graph Network) model for provenance graph attack detection.

Maintains per-entity state vectors (memories) updated via attention when entities
participate in events. For each edge, classifies attack/benign using:
    [mem_src, mem_dst, bert_emb_src, bert_emb_dst, edge_emb, time_enc_src, time_enc_dst]

Memory updates use self-attention over 4 feature tokens (no GRU/LSTM):
    [current_memory, interaction_message, entity_embedding, time_encoding]
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class TGNTimeEncoder(nn.Module):
    """Encode time deltas: log1p(delta_seconds) -> Linear -> cos()."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.lin = nn.Linear(1, time_dim)

    def forward(self, delta_ns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_ns: [N] raw timestamp deltas in nanoseconds (int64 or float).
        Returns:
            [N, time_dim] encoded time features.
        """
        delta_sec = delta_ns.float() / 1e9
        x = torch.log1p(delta_sec.abs()).unsqueeze(-1)  # [N, 1]
        return self.lin(x).cos()  # [N, time_dim]


class TGNMessageFunction(nn.Module):
    """Compute interaction message from node memories, embeddings, edge type, and time.

    Input: concat(self_mem, other_mem, self_emb, other_emb, edge_emb, time_enc)
    Output: [N, memory_dim] message vector.
    """

    def __init__(self, memory_dim: int, entity_emb_dim: int, edge_emb_dim: int, time_dim: int):
        super().__init__()
        input_dim = 2 * memory_dim + 2 * entity_emb_dim + edge_emb_dim + time_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.ReLU(),
        )

    def forward(self, self_mem, other_mem, self_emb, other_emb, edge_emb, time_enc):
        """All inputs: [N, respective_dim]. Returns: [N, memory_dim]."""
        return self.proj(torch.cat([self_mem, other_mem, self_emb, other_emb, edge_emb, time_enc], dim=-1))


class AttentionMemoryUpdater(nn.Module):
    """Update entity memory via self-attention over 4 feature tokens.

    Tokens:
        0: current entity memory (output here = updated memory)
        1: aggregated interaction message
        2: entity's static BERT embedding (identity)
        3: time delta encoding (temporal context)

    Each token is projected to memory_dim, augmented with learned token-type
    embeddings, then processed by a TransformerEncoderLayer (self-attention +
    FFN with pre-norm and GELU). Output at position 0 is the new memory.
    """

    NUM_TOKEN_TYPES = 4

    def __init__(self, memory_dim: int, entity_emb_dim: int, time_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Project heterogeneous features to uniform memory_dim
        self.proj_mem = nn.Linear(memory_dim, memory_dim)
        self.proj_msg = nn.Linear(memory_dim, memory_dim)  # message is already memory_dim
        self.proj_emb = nn.Linear(entity_emb_dim, memory_dim)
        self.proj_time = nn.Linear(time_dim, memory_dim)

        # Token-type embeddings so the model distinguishes the 4 slots
        self.token_type_emb = nn.Embedding(self.NUM_TOKEN_TYPES, memory_dim)

        # Self-attention encoder (1 layer, pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=num_heads,
            dim_feedforward=memory_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_norm = nn.LayerNorm(memory_dim)

    def forward(self, memory, message, entity_emb, time_enc):
        """
        Args:
            memory:     [B, memory_dim]
            message:    [B, memory_dim]
            entity_emb: [B, entity_emb_dim]
            time_enc:   [B, time_dim]
        Returns:
            [B, memory_dim] updated memory.
        """
        B = memory.size(0)
        device = memory.device

        # Project each feature to memory_dim
        tok0 = self.proj_mem(memory)       # [B, D]
        tok1 = self.proj_msg(message)      # [B, D]
        tok2 = self.proj_emb(entity_emb)   # [B, D]
        tok3 = self.proj_time(time_enc)    # [B, D]

        tokens = torch.stack([tok0, tok1, tok2, tok3], dim=1)  # [B, 4, D]

        # Add token-type embeddings
        type_ids = torch.arange(self.NUM_TOKEN_TYPES, device=device)  # [4]
        tokens = tokens + self.token_type_emb(type_ids).unsqueeze(0)  # broadcast [1, 4, D]

        # Self-attention
        out = self.encoder(tokens)  # [B, 4, D]
        return self.out_norm(out[:, 0])  # [B, D] — position 0 = updated memory


class EdgeTypeClassifier(nn.Module):
    """Static edge type classifier from pre-computed entity embeddings.

    Given a (src, dst) pair, predicts the edge type from concatenated BERT
    entity embeddings.  No temporal memory, no state — each edge is scored
    independently.

    Training: CE loss on predicted edge type.
    Inference: per-edge CE loss as anomaly score.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        entity_emb_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.entity_emb_dim = entity_emb_dim

        self.register_buffer("entity_embs", torch.zeros(num_nodes, entity_emb_dim))

        num_classes = num_edge_types + 1
        self.classifier = nn.Sequential(
            nn.Linear(2 * entity_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def set_entity_embeddings(self, embs: torch.Tensor):
        self.entity_embs.copy_(embs.to(self.entity_embs.device))

    def predict(self, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        """Predict edge type logits.

        Args:
            src_idx: [B] int64 contiguous node indices.
            dst_idx: [B] int64 contiguous node indices.
        Returns:
            [B, num_classes] raw logits.
        """
        features = torch.cat([
            self.entity_embs[src_idx],
            self.entity_embs[dst_idx],
        ], dim=-1)
        return self.classifier(features)


class ProvenanceTGN(nn.Module):
    """TGN-based anomaly detector for provenance graphs.

    Components:
        entity_embs:    [N, H] pre-computed frozen BERT embeddings (buffer)
        memory:         [N, memory_dim] evolving entity state vectors (buffer)
        last_update:    [N] last-update timestamps in nanoseconds (buffer)
        node_types:     [N] int node type IDs (buffer, optional)
        edge_emb:       Embedding(num_edge_types+1, edge_emb_dim)
        time_enc:       TGNTimeEncoder(time_dim)
        message_fn:     TGNMessageFunction → message vectors for memory update
        memory_updater: AttentionMemoryUpdater (self-attention, no GRU)
        classifier:     MLP → binary anomaly logit

    Optional classifier inputs (controlled by flags):
        use_node_type_emb: include learned src/dst node-type embeddings so the
            classifier can learn type-specific score baselines.
        use_time_emb: include src/dst time-delta encodings in the classifier.
    """

    NUM_NODE_TYPES = 4  # 0=unknown, 1=subject, 2=file, 3=netflow

    def __init__(
        self,
        num_nodes: int,
        num_edge_types: int,
        entity_emb_dim: int,
        memory_dim: int = 128,
        edge_emb_dim: int = 128,
        time_dim: int = 16,
        num_heads: int = 4,
        use_node_type_emb: bool = False,
        use_time_emb: bool = True,
        use_memory: bool = True,
        use_entity_emb: bool = True,
        use_event_bert_emb: bool = False,
        event_bert_emb_dim: int = 0,
        score_gated_memory: bool = False,
        temporal_decay: bool = False,
        anomaly_accumulator: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.entity_emb_dim = entity_emb_dim
        self.use_node_type_emb = use_node_type_emb
        self.use_time_emb = use_time_emb
        self.use_memory = use_memory
        self.use_entity_emb = use_entity_emb
        self.use_event_bert_emb = use_event_bert_emb
        self.event_bert_emb_dim = event_bert_emb_dim
        self.score_gated_memory = score_gated_memory
        self.temporal_decay = temporal_decay
        self.anomaly_accumulator = anomaly_accumulator
        self.device = device

        # ── Buffers (non-parameter state) ──────────────────────────
        self.register_buffer("entity_embs", torch.zeros(num_nodes, entity_emb_dim))
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("node_types", torch.zeros(num_nodes, dtype=torch.long))
        if anomaly_accumulator:
            self.register_buffer("anomaly_accum", torch.zeros(num_nodes))

        # Differentiable memory overlay for training (truncated BPTT depth 1).
        # When set, predict/classify read from this instead of the buffer.
        self._diff_memory = None

        # ── Learnable modules ──────────────────────────────────────
        self.edge_emb = nn.Embedding(num_edge_types + 1, edge_emb_dim, padding_idx=0)
        self.time_enc = TGNTimeEncoder(time_dim)

        self.message_fn = TGNMessageFunction(
            memory_dim, entity_emb_dim, edge_emb_dim, time_dim,
        )

        self.memory_updater = AttentionMemoryUpdater(
            memory_dim, entity_emb_dim, time_dim, num_heads=num_heads,
        )

        # ── Score-gated memory: learned gate from anomaly score ───
        if score_gated_memory:
            self.score_gate = nn.Sequential(
                nn.Linear(1, memory_dim),
                nn.Sigmoid(),
            )

        # ── Temporal decay: learned base decay rate ───────────────
        if temporal_decay:
            self.decay_rate = nn.Parameter(torch.zeros(1))

        # ── Optional node-type embeddings ──────────────────────────
        type_emb_dim = edge_emb_dim  # reuse edge_emb_dim for consistency
        if use_node_type_emb:
            self.node_type_emb = nn.Embedding(self.NUM_NODE_TYPES, type_emb_dim)
        else:
            type_emb_dim = 0

        # ── Classifier (contrastive objective) ─────────────────────
        # Input: [emb_src, emb_dst, edge_emb] + optional [mem, time, type, event_bert]
        cls_input_dim = edge_emb_dim
        if use_entity_emb:
            cls_input_dim += 2 * entity_emb_dim
        if use_memory:
            cls_input_dim += 2 * memory_dim
        if use_time_emb:
            cls_input_dim += 2 * time_dim
        if use_node_type_emb:
            cls_input_dim += 2 * type_emb_dim
        if use_event_bert_emb:
            cls_input_dim += 2 * event_bert_emb_dim  # paired src + dst
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, 1),
        )

        # ── Edge type predictor (edge_pred objective) ─────────────
        # Same features as classifier but WITHOUT edge_emb (that's the target).
        pred_input_dim = 0
        if use_entity_emb:
            pred_input_dim += 2 * entity_emb_dim
        if use_memory:
            pred_input_dim += 2 * memory_dim
        if use_time_emb:
            pred_input_dim += 2 * time_dim
        if use_node_type_emb:
            pred_input_dim += 2 * type_emb_dim
        if use_event_bert_emb:
            pred_input_dim += 2 * event_bert_emb_dim  # paired src + dst
        if pred_input_dim == 0:
            raise ValueError(
                "Edge predictor has 0 input features. Enable at least one of: "
                "use_memory, use_entity_emb, use_time_emb, use_node_type_emb, use_event_bert_emb"
            )
        num_edge_classes = num_edge_types + 1  # match edge_emb vocabulary
        self.edge_predictor = nn.Sequential(
            nn.Linear(pred_input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, num_edge_classes),
        )

        # ── Edge embedding predictor (edge_pred_emb objective) ─────
        # Same input features as edge_predictor, but predicts the continuous
        # edge embedding vector instead of the discrete type.
        self.edge_emb_predictor = nn.Sequential(
            nn.Linear(pred_input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, edge_emb_dim),
        )

        # ── Node type predictor (node_pred objective) ─────────────
        # Predicts dst node type from edge_emb + memory + entity_emb + time_emb.
        # Excludes node_type_emb (that's the prediction target).
        # Only uses event_src_emb (NOT event_dst_emb) because the dst span
        # directly contains the dst_type token which IS the prediction target.
        node_pred_input_dim = edge_emb_dim  # always include edge embedding
        if use_entity_emb:
            node_pred_input_dim += 2 * entity_emb_dim
        if use_memory:
            node_pred_input_dim += 2 * memory_dim
        if use_time_emb:
            node_pred_input_dim += 2 * time_dim
        if use_event_bert_emb:
            node_pred_input_dim += event_bert_emb_dim  # src only (dst leaks target)
        self.node_predictor = nn.Sequential(
            nn.Linear(node_pred_input_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim, self.NUM_NODE_TYPES),
        )

    def _get_memory(self):
        """Return current effective memory: differentiable overlay if set, else buffer."""
        return self._diff_memory if self._diff_memory is not None else self.memory

    # ── State management ───────────────────────────────────────────

    def set_entity_embeddings(self, embs: torch.Tensor):
        """Set pre-computed BERT entity embeddings (called once before training)."""
        self.entity_embs.copy_(embs.to(self.entity_embs.device))

    def set_node_types(self, types: torch.Tensor):
        """Set node type IDs (0=unknown, 1=subject, 2=file, 3=netflow)."""
        self.node_types.copy_(types.to(self.node_types.device))

    def reset_memory(self):
        """Zero all entity memories and last-update timestamps."""
        self.memory.zero_()
        self.last_update.zero_()
        self._diff_memory = None
        if self.anomaly_accumulator:
            self.anomaly_accum.zero_()

    def get_memory_state(self) -> dict:
        """Return a detached copy of current memory state for checkpointing."""
        mem = self._get_memory()
        state = {
            "memory": mem.detach().clone().cpu(),
            "last_update": self.last_update.detach().clone().cpu(),
        }
        if self.anomaly_accumulator:
            state["anomaly_accum"] = self.anomaly_accum.detach().clone().cpu()
        return state

    def load_memory_state(self, state: dict):
        """Restore memory state from a saved dict."""
        self.memory.copy_(state["memory"].to(self.memory.device))
        self.last_update.copy_(state["last_update"].to(self.last_update.device))
        self._diff_memory = None
        if self.anomaly_accumulator and "anomaly_accum" in state:
            self.anomaly_accum.copy_(state["anomaly_accum"].to(self.anomaly_accum.device))

    # ── Classification ─────────────────────────────────────────────

    def classify_edges(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        event_src_emb: torch.Tensor = None,
        event_dst_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute anomaly logits for a batch of edges using current memories.

        Args:
            src_idx:    [B] int64 contiguous node indices
            dst_idx:    [B] int64 contiguous node indices
            edge_types: [B] int64 edge type IDs
            timestamps: [B] int64 timestamps in nanoseconds
            event_src_emb: [B, H] optional per-edge contextual src embeddings
            event_dst_emb: [B, H] optional per-edge contextual dst embeddings

        Returns:
            [B, 1] raw logits (before sigmoid).
        """
        e_emb = self.edge_emb(edge_types)
        parts = [e_emb]

        if self.use_entity_emb:
            parts.extend([self.entity_embs[src_idx], self.entity_embs[dst_idx]])

        if self.use_memory:
            mem = self._get_memory()
            parts.extend([mem[src_idx], mem[dst_idx]])

        if self.use_time_emb:
            src_dt = timestamps - self.last_update[src_idx]
            dst_dt = timestamps - self.last_update[dst_idx]
            parts.extend([self.time_enc(src_dt), self.time_enc(dst_dt)])

        if self.use_node_type_emb:
            parts.extend([
                self.node_type_emb(self.node_types[src_idx]),
                self.node_type_emb(self.node_types[dst_idx]),
            ])

        if self.use_event_bert_emb and self.event_bert_emb_dim > 0 and event_src_emb is not None:
            parts.extend([event_src_emb, event_dst_emb])

        return self.classifier(torch.cat(parts, dim=-1))

    # ── Edge type prediction ─────────────────────────────────────

    def predict_edge_types(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        timestamps: torch.Tensor,
        event_src_emb: torch.Tensor = None,
        event_dst_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict edge type logits from memories and embeddings (no edge type input).

        Args:
            src_idx:    [B] int64 contiguous node indices
            dst_idx:    [B] int64 contiguous node indices
            timestamps: [B] int64 timestamps in nanoseconds
            event_src_emb: [B, H] optional per-edge contextual src embeddings
            event_dst_emb: [B, H] optional per-edge contextual dst embeddings

        Returns:
            [B, num_edge_classes] raw logits.
        """
        parts = []

        if self.use_entity_emb:
            parts.extend([self.entity_embs[src_idx], self.entity_embs[dst_idx]])

        if self.use_memory:
            mem = self._get_memory()
            parts.extend([mem[src_idx], mem[dst_idx]])

        if self.use_time_emb:
            src_dt = timestamps - self.last_update[src_idx]
            dst_dt = timestamps - self.last_update[dst_idx]
            parts.extend([self.time_enc(src_dt), self.time_enc(dst_dt)])

        if self.use_node_type_emb:
            parts.extend([
                self.node_type_emb(self.node_types[src_idx]),
                self.node_type_emb(self.node_types[dst_idx]),
            ])

        if self.use_event_bert_emb and self.event_bert_emb_dim > 0 and event_src_emb is not None:
            parts.extend([event_src_emb, event_dst_emb])

        return self.edge_predictor(torch.cat(parts, dim=-1))

    # ── Edge embedding prediction ─────────────────────────────────

    def predict_edge_embedding(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        timestamps: torch.Tensor,
        event_src_emb: torch.Tensor = None,
        event_dst_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict edge embedding vector from memories and embeddings.

        Uses the same input features as predict_edge_types but outputs a
        continuous embedding vector [B, edge_emb_dim] instead of class logits.
        """
        parts = []

        if self.use_entity_emb:
            parts.extend([self.entity_embs[src_idx], self.entity_embs[dst_idx]])

        if self.use_memory:
            mem = self._get_memory()
            parts.extend([mem[src_idx], mem[dst_idx]])

        if self.use_time_emb:
            src_dt = timestamps - self.last_update[src_idx]
            dst_dt = timestamps - self.last_update[dst_idx]
            parts.extend([self.time_enc(src_dt), self.time_enc(dst_dt)])

        if self.use_node_type_emb:
            parts.extend([
                self.node_type_emb(self.node_types[src_idx]),
                self.node_type_emb(self.node_types[dst_idx]),
            ])

        if self.use_event_bert_emb and self.event_bert_emb_dim > 0 and event_src_emb is not None:
            parts.extend([event_src_emb, event_dst_emb])

        return self.edge_emb_predictor(torch.cat(parts, dim=-1))

    # ── Node type prediction ──────────────────────────────────────

    def predict_node_types(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        event_src_emb: torch.Tensor = None,
        event_dst_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict destination node type logits.

        Uses edge_emb + memory + entity_emb + time_emb (no node_type_emb,
        since node type is the prediction target).

        Args:
            src_idx:    [B] int64 contiguous node indices
            dst_idx:    [B] int64 contiguous node indices
            edge_types: [B] int64 edge type IDs
            timestamps: [B] int64 timestamps in nanoseconds
            event_src_emb: [B, H] optional per-edge contextual src embeddings
            event_dst_emb: [B, H] optional per-edge contextual dst embeddings

        Returns:
            [B, NUM_NODE_TYPES] raw logits for destination node type.
        """
        e_emb = self.edge_emb(edge_types)
        parts = [e_emb]

        if self.use_entity_emb:
            parts.extend([self.entity_embs[src_idx], self.entity_embs[dst_idx]])

        if self.use_memory:
            mem = self._get_memory()
            parts.extend([mem[src_idx], mem[dst_idx]])

        if self.use_time_emb:
            src_dt = timestamps - self.last_update[src_idx]
            dst_dt = timestamps - self.last_update[dst_idx]
            parts.extend([self.time_enc(src_dt), self.time_enc(dst_dt)])

        if self.use_event_bert_emb and self.event_bert_emb_dim > 0 and event_src_emb is not None:
            # Only event_src_emb — event_dst_emb encodes dst_type (the target)
            parts.append(event_src_emb)

        return self.node_predictor(torch.cat(parts, dim=-1))

    # ── Memory update ──────────────────────────────────────────────

    def _compute_update(
        self,
        mem: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        edge_scores: torch.Tensor = None,
    ):
        """Core memory update computation (shared by diff and inplace paths).

        Steps:
          1. Compute asymmetric messages for src and dst.
          2. (score_gated_memory) Weight messages by learned gate on edge_scores.
          3. Scatter-mean aggregate per unique node.
          4. (temporal_decay) Decay old memory toward zero based on time elapsed.
             If anomaly_accumulator is also active, nodes with high accumulated
             anomaly decay slower (stickier memory).
          5. Run attention memory updater on (decayed) memory + aggregated messages.
          6. (anomaly_accumulator) Update per-node anomaly EMA under no_grad.

        Returns: (new_mem, unique_nodes, all_idx, all_ts)
        """
        e_emb = self.edge_emb(edge_types)

        src_dt = timestamps - self.last_update[src_idx]
        dst_dt = timestamps - self.last_update[dst_idx]
        src_time = self.time_enc(src_dt)
        dst_time = self.time_enc(dst_dt)

        # Messages: asymmetric (self, other) perspective
        src_msg = self.message_fn(
            mem[src_idx], mem[dst_idx],
            self.entity_embs[src_idx], self.entity_embs[dst_idx],
            e_emb, src_time,
        )
        dst_msg = self.message_fn(
            mem[dst_idx], mem[src_idx],
            self.entity_embs[dst_idx], self.entity_embs[src_idx],
            e_emb, dst_time,
        )

        # Score-gated messages: weight by anomaly score before aggregation
        if self.score_gated_memory and edge_scores is not None:
            gate = self.score_gate(edge_scores.unsqueeze(-1))  # [B, memory_dim]
            src_msg = src_msg * gate
            dst_msg = dst_msg * gate

        # Scatter-mean aggregate per unique node
        all_idx = torch.cat([src_idx, dst_idx])           # [2B]
        all_msg = torch.cat([src_msg, dst_msg])            # [2B, memory_dim]
        all_time_enc = torch.cat([src_time, dst_time])     # [2B, time_dim]
        all_ts = torch.cat([timestamps, timestamps])       # [2B]

        aggr_msg = scatter(all_msg, all_idx, dim=0,
                           dim_size=self.num_nodes, reduce="mean")
        aggr_time = scatter(all_time_enc, all_idx, dim=0,
                            dim_size=self.num_nodes, reduce="mean")

        unique_nodes = all_idx.unique()

        # Temporal decay: shrink old memory toward zero based on time elapsed
        if self.temporal_decay:
            node_max_ts = scatter(all_ts, all_idx, dim=0,
                                  dim_size=self.num_nodes, reduce="max")
            node_dt = node_max_ts[unique_nodes] - self.last_update[unique_nodes]
            dt_sec = node_dt.float() / 1e9
            base_rate = torch.nn.functional.softplus(self.decay_rate)

            if self.anomaly_accumulator:
                # High accumulated anomaly → slower decay (stickier memory)
                accum = self.anomaly_accum[unique_nodes].unsqueeze(-1)
                effective_rate = base_rate / (1.0 + accum)
            else:
                effective_rate = base_rate

            decay = torch.exp(-effective_rate * torch.log1p(dt_sec.abs()).unsqueeze(-1))
            node_mem = mem[unique_nodes] * decay
        else:
            node_mem = mem[unique_nodes]

        new_mem = self.memory_updater(
            node_mem,
            aggr_msg[unique_nodes],
            self.entity_embs[unique_nodes],
            aggr_time[unique_nodes],
        )

        # Update anomaly accumulator (always non-differentiable)
        if self.anomaly_accumulator and edge_scores is not None:
            with torch.no_grad():
                all_scores = torch.cat([edge_scores, edge_scores])
                node_scores = scatter(all_scores.float(), all_idx, dim=0,
                                      dim_size=self.num_nodes, reduce="mean")
                self.anomaly_accum[unique_nodes] = (
                    0.95 * self.anomaly_accum[unique_nodes]
                    + 0.05 * node_scores[unique_nodes]
                )

        return new_mem, unique_nodes, all_idx, all_ts

    def update_memory(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        edge_scores: torch.Tensor = None,
    ):
        """Update memories for nodes involved in edges.

        Args:
            edge_scores: [B] per-edge anomaly scores (detached). Used by
                score_gated_memory to weight messages. Ignored if None.
        """
        if self.training:
            self._update_memory_diff(src_idx, dst_idx, edge_types, timestamps, edge_scores)
        else:
            self._update_memory_inplace(src_idx, dst_idx, edge_types, timestamps, edge_scores)

    def _update_memory_diff(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        edge_scores: torch.Tensor = None,
    ):
        """Differentiable memory update for training (creates _diff_memory overlay)."""
        device = src_idx.device
        mem = self._get_memory()

        new_mem, unique_nodes, all_idx, all_ts = self._compute_update(
            mem, src_idx, dst_idx, edge_types, timestamps, edge_scores,
        )

        # Build differentiable overlay: new_mem at updated positions,
        # detached buffer elsewhere.
        new_mem_full = scatter(new_mem, unique_nodes, dim=0,
                               dim_size=self.num_nodes, reduce="sum")
        mask = torch.zeros(self.num_nodes, 1, device=device)
        mask[unique_nodes] = 1.0
        self._diff_memory = new_mem_full + self.memory.detach() * (1 - mask)

        # Update last_update timestamps (non-differentiable)
        with torch.no_grad():
            max_ts = scatter(all_ts, all_idx, dim=0,
                             dim_size=self.num_nodes, reduce="max")
            new_ts = max_ts[unique_nodes]
            ts_mask = new_ts > self.last_update[unique_nodes]
            self.last_update[unique_nodes] = torch.where(
                ts_mask, new_ts, self.last_update[unique_nodes],
            )

    def _update_memory_inplace(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_types: torch.Tensor,
        timestamps: torch.Tensor,
        edge_scores: torch.Tensor = None,
    ):
        """In-place memory update for inference (no gradient tracking)."""
        with torch.no_grad():
            new_mem, unique_nodes, all_idx, all_ts = self._compute_update(
                self.memory, src_idx, dst_idx, edge_types, timestamps, edge_scores,
            )

            self.memory[unique_nodes] = new_mem.detach()

            max_ts = scatter(all_ts, all_idx, dim=0,
                             dim_size=self.num_nodes, reduce="max")
            new_ts = max_ts[unique_nodes]
            mask = new_ts > self.last_update[unique_nodes]
            self.last_update[unique_nodes] = torch.where(
                mask, new_ts, self.last_update[unique_nodes],
            )

    def detach_memory(self):
        """Sync differentiable memory overlay to buffer and clear it.

        Call after loss.backward() + opt.step() and before update_memory()
        during training.  This breaks the computation graph between batches,
        implementing truncated BPTT (depth 1).
        """
        if self._diff_memory is not None:
            self.memory.copy_(self._diff_memory.detach())
            self._diff_memory = None
