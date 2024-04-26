from provnet_utils import *
from losses import sce_loss
from config import *
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: refactor
max_node_num = 967390
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, node_dropout):
        super(GraphAttentionEmbedding, self).__init__()
        self.conv = TransformerConv(in_channels, hid_channels, heads=8, dropout=node_dropout)
        self.conv2 = TransformerConv(hid_channels * 8, out_channels, heads=1, concat=False, dropout=node_dropout)

        self.dropout = nn.Dropout(node_dropout)

    def forward(self, x, edge_index):
        x = x.to(device)
        x = F.relu(self.conv(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        return x

# class NodeRecon(torch.nn.Module):
#     def __init__(self, layer1_in=64, layer2_in=80, layer3_in=100, layer3_out=128):
#         super(NodeRecon, self).__init__()
#         self.conv = TransformerConv(layer1_in, layer2_in, heads=8, concat=False, dropout=0.0)
#         self.conv2 = TransformerConv(layer2_in, layer3_in, heads=8, concat=False, dropout=0.0)
#         self.conv3 = TransformerConv(layer3_in, layer3_out, heads=8, concat=False, dropout=0.0)

#     def forward(self, x, edge_index):
#         x = x.to(device)
#         x = F.relu(self.conv(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = torch.tanh(self.conv3(x, edge_index))
#         return x

class NodeRecon_MLP(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, use_bias):
        super(NodeRecon_MLP, self).__init__()
        self.conv = nn.Linear(in_dim, h_dim, bias=use_bias)
        self.conv2 = nn.Linear(h_dim, out_dim, bias=use_bias)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv(x))
        x = torch.tanh(self.conv2(x))
        return x

class TGNEncoder(torch.nn.Module):
    def __init__(self, encoder, memory, neighbor_loader):
        super(TGNEncoder, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.neighbor_loader = neighbor_loader

    def forward(self, edge_index, t, msg, inference=False):
        src, dst = edge_index
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        h, last_update = self.memory(n_id)
        h = self.encoder(h, edge_index)

        # Decoding
        h_src = h[assoc[src]]
        h_dst = h[assoc[dst]]

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
            h_src, h_dst = self.encoder(edge_index, t, msg, inference=inference)
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
