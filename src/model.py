from provnet_utils import *
from losses import sce_loss
from config import *
import torch.nn as nn


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, edge_dim, node_dropout, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        
        self.time_enc = time_enc
        
        self.conv = TransformerConv(in_channels, hid_channels, heads=8, dropout=node_dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hid_channels * 8, out_channels, heads=1, concat=False, dropout=node_dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(node_dropout)

    def forward(self, x, edge_index, edge_feats=None):
        x = F.relu(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_feats))
        return x

class NodeRecon_MLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, use_bias):
        super(NodeRecon_MLP, self).__init__()
        self.conv = nn.Linear(in_dim, h_dim, bias=use_bias)
        self.conv2 = nn.Linear(h_dim, out_dim, bias=use_bias)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.tanh(self.conv2(x))
        return x

class TGNEncoder(torch.nn.Module):
    def __init__(self, encoder, memory, neighbor_loader, use_time_encoding):
        super(TGNEncoder, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.neighbor_loader = neighbor_loader
        self.use_time_encoding = use_time_encoding
        self.device = self.memory.memory.device
        self.assoc = torch.empty(self.memory.num_nodes, dtype=torch.long, device=self.device)

    def forward(self, edge_index, t, msg, inference=False, **kwargs):
        src, dst = edge_index
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        # Get updated memory of all nodes involved in the computation.
        h, last_update = self.memory(n_id)
        
        # Call the downstream encoder with possibly edge features
        edge_feats = []
        if self.use_edge_feats:
            edge_feats.append(msg)
        if self.use_time_encoding:
            rel_t = last_update[edge_index[0]] - t
            rel_t_enc = self.time_enc(rel_t.to(h.dtype))
            edge_feats.append(rel_t_enc)
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        h = self.encoder(h, edge_index, edge_feats=edge_feats)

        # Decoding
        h_src = h[self.assoc[src]]
        h_dst = h[self.assoc[dst]]

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)
        
        # Detaching memory is only useful for backprop in training
        if not inference:
            self.memory.detach()
        
        return h_src, h_dst

    def reset_state(self):
        self.memory.reset_state()  # Flushes memory.
        self.neighbor_loader.reset_state()  # Empties the graph.

class Model(torch.nn.Module):
    def __init__(self, encoder, decoder, losses: str):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
        self.losses = set(losses.split(","))
        for l in self.losses:
            if l not in ["recons_msg"]:
                raise ValueError(f"Invalid loss function {l}")
        
    def forward(self, edge_index, t, x_src, x_dst, msg, inference=False):
        train_mode = not inference
        
        with torch.set_grad_enabled(train_mode):
            h_src, h_dst = self.encoder(edge_index=edge_index, t=t, x=(x_src, x_dst), msg=msg, inference=inference)
            x_src_hat, x_dst_hat = self.decoder(h_src, h_dst)
            
            # Train mode: loss | Inference mode: edge scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)

            if "recons_msg" in self.losses:
                loss_src = sce_loss(x_src_hat, x_src, inference=inference)
                loss_dst = sce_loss(x_dst_hat, x_dst, inference=inference)

                loss_or_scores = loss_or_scores + (loss_src + loss_dst)
                
            return loss_or_scores


class SrcDstNodeDecoder(torch.nn.Module):
    def __init__(self, src_decoder, dst_decoder):
        super(SrcDstNodeDecoder, self).__init__()
        self.src_decoder = src_decoder
        self.dst_decoder = dst_decoder

    def forward(self, h_src, h_dst):
        return self.src_decoder(h_src), self.dst_decoder(h_dst)
