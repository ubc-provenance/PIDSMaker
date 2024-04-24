from provnet_utils import *
from config import *
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: refactor
max_node_num = 967390
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

criterion = sce_loss


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


def cal_node_loss(y_pred, y_true):
    loss = []
    for i in range(len(y_pred)):
        src_dst_loss = criterion(y_pred[i], y_true[i])

        loss.append(src_dst_loss)
    return torch.tensor(loss)
