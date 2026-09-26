import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictEdgeSupervised(nn.Module):
    """Binary supervised edge classification: benign (0) vs attack (1).

    Input to the decoder per edge: [x_src | edge_type | x_dst | edge_type]
    (edge_type one-hot appended to both node embeddings symmetrically).

    Uses raw SPIDER node embeddings (batch.x_src, batch.x_dst) directly,
    bypassing the GNN encoder, so that attack features extracted from val/test
    graphs and benign features from training graphs share the same embedding
    space.

    During training, attack edges are oversampled 1:1 with replacement to match
    the benign batch size.  This counters the extreme class imbalance (typically
    O(20) attacks vs O(millions) benign edges) without any modification to the
    data loading pipeline.

    At inference time, per-edge BCE losses are returned as anomaly scores.
    """

    def __init__(
        self,
        decoder: nn.Module,
        atk_x_src: torch.Tensor,
        atk_x_dst: torch.Tensor,
        atk_edge_type: torch.Tensor,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("atk_x_src", atk_x_src)
        self.register_buffer("atk_x_dst", atk_x_dst)
        self.register_buffer("atk_edge_type", atk_edge_type)
        self.pos_weight_val = pos_weight

    @staticmethod
    def _augment(x_src, x_dst, edge_type):
        """Append edge_type one-hot to both node embeddings."""
        return (
            torch.cat([x_src, edge_type], dim=-1),
            torch.cat([x_dst, edge_type], dim=-1),
        )

    def forward(self, edge_type, inference: bool, batch, **kwargs):
        x_src, x_dst = self._augment(batch.x_src, batch.x_dst, batch.edge_type)
        device = x_src.device

        if inference:
            logits = self.decoder(h_src=x_src, h_dst=x_dst).squeeze(-1)
            scores = F.binary_cross_entropy_with_logits(
                logits, batch.y.float(), reduction="none"
            )
            return {"loss": scores}

        # Training: oversample attack edges 1:1 with replacement
        n = x_src.shape[0]
        n_atk = self.atk_x_src.shape[0]
        idx = torch.randint(0, n_atk, (n,), device=device)
        atk_src, atk_dst = self._augment(
            self.atk_x_src[idx], self.atk_x_dst[idx], self.atk_edge_type[idx]
        )

        all_src = torch.cat([x_src, atk_src], dim=0)
        all_dst = torch.cat([x_dst, atk_dst], dim=0)
        labels = torch.cat([
            torch.zeros(n, device=device),
            torch.ones(n, device=device),
        ])

        logits = self.decoder(h_src=all_src, h_dst=all_dst).squeeze(-1)
        pw = torch.tensor([self.pos_weight_val], device=device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        return {"loss": loss}
