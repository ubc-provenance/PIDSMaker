from provnet_utils import *
from config import *
import torch.nn as nn


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, node_dropout):
        super(GraphAttentionEmbedding, self).__init__()
        
        self.conv = TransformerConv(in_dim, hid_dim, heads=8, dropout=node_dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hid_dim * 8, out_dim, heads=1, concat=False, dropout=node_dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(node_dropout)

    def forward(self, x, edge_index, edge_feats=None):
        x = F.relu(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_feats))
        return x

class NodeRecon_MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, use_bias):
        super(NodeRecon_MLP, self).__init__()
        self.conv = nn.Linear(in_dim, h_dim, bias=use_bias)
        self.conv2 = nn.Linear(h_dim, out_dim, bias=use_bias)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.tanh(self.conv2(x))
        return x

class TGNEncoder(nn.Module):
    def __init__(self, encoder, memory, neighbor_loader, time_encoder, use_time_encoding, use_msg_as_edge_feature):
        super(TGNEncoder, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.neighbor_loader = neighbor_loader
        self.use_time_encoding = use_time_encoding
        self.use_msg_as_edge_feature = use_msg_as_edge_feature
        self.device = self.memory.memory.device
        self.assoc = torch.empty(self.memory.num_nodes, dtype=torch.long, device=self.device)
        
        if self.use_time_encoding:
            self.time_encoder = time_encoder

    def forward(self, edge_index, t, msg, full_data, inference=False, **kwargs):
        src, dst = edge_index
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        
        curr_msg = full_data.msg[e_id]
        curr_t = full_data.t[e_id]

        # Get updated memory of all nodes involved in the computation.
        h, last_update = self.memory(n_id)
        
        # Call the downstream encoder with possibly edge features
        edge_feats = []
        if self.use_msg_as_edge_feature:
            edge_feats.append(curr_msg)
        if self.use_time_encoding:
            rel_t = last_update[edge_index[0]] - curr_t
            rel_t_enc = self.time_encoder(rel_t.to(h.dtype))
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

class Model(nn.Module):
    def __init__(self, encoder: nn.Module, decoders: list[nn.Module]):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = decoders
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference
        
        with torch.set_grad_enabled(train_mode):
            edge_index = torch.stack([batch.src, batch.dst])
            h_src, h_dst = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=(batch.x_src, batch.x_dst),
                msg=batch.msg,
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
            )
            
            # Train mode: loss | Inference mode: edge scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)

            for decoder in self.decoders:
                loss = decoder(h_src, h_dst, edge_index=edge_index, edge_type=batch.edge_type, inference=inference)
                loss_or_scores = loss_or_scores + loss
                
            return loss_or_scores


class SrcDstNodeDecoder(nn.Module):
    def __init__(self, src_decoder, dst_decoder, loss_fn):
        super(SrcDstNodeDecoder, self).__init__()
        self.src_decoder = src_decoder
        self.dst_decoder = dst_decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, inference, **kwargs):
        x_src_hat, x_dst_hat = self.src_decoder(h_src), self.dst_decoder(h_dst)
        
        loss_src = self.loss_fn(x_src_hat, x_src, inference=inference)
        loss_dst = self.loss_fn(x_dst_hat, x_dst, inference=inference)

        return loss_src + loss_dst

class EdgeTypeDecoder(nn.Module):
    def __init__(self, in_dim, num_edge_types, loss_fn):
        super(EdgeTypeDecoder, self).__init__()
        self.lin_src = Linear(in_dim, in_dim*2)
        self.lin_dst = Linear(in_dim, in_dim*2)
        
        self.lin_seq = nn.Sequential(
            Linear(in_dim*4, in_dim*8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_dim*8, in_dim*2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_dim*2, int(in_dim//2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_dim//2), num_edge_types)                   
        )
        self.loss_fn = loss_fn
        
    def forward(self, h_src, h_dst, edge_type, **kwargs):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)      
        h = self.lin_seq (h)
        
        loss = self.loss_fn(h, edge_type)
        return loss
