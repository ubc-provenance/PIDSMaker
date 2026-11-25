"""TextRCNN over the sequence of the last snapshot embeddings (ProGrapher's anomaly detector).

Each graph (time window) is a snapshot embedded as a whole by graph2vec (`batch.graph_emb`, see
`featurization.used_method=graph2vec`). Like `snapshot_gru`, the encoder keeps a state across
graphs: the embeddings of the last `sequence_length` snapshots seen, in the order of the graphs
(chronological, as graphs are iterated by time window). From this sequence, it predicts the
embedding of the current snapshot, which the `predict_snapshot_embedding` objective compares with
its actual embedding.

The TextRCNN of ProGrapher (Lai et al., AAAI 2015) represents each snapshot S_i of the sequence by
x_i = [left(S_i); E(S_i); right(S_i)], the left and right contexts being computed by a
bidirectional RNN, projects it with y_i = tanh(W x_i + b), max-pools the y_i over the sequence,
and maps the result to the predicted embedding with a fully-connected layer.

The state is filled with the actual embeddings of the snapshots (teacher forcing), reset before
each training epoch, and carried from the training graphs to the validation and test graphs, as
inference follows training. With fewer previous snapshots than `sequence_length` (e.g. at the
beginning of training), the sequence is shorter; the first snapshot is predicted from a zero
embedding.
"""

import torch
import torch.nn as nn


class TextRCNN(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_layers, sequence_length, dropout=0.0, rnn_type="lstm"):
        super().__init__()
        rnn_classes = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        if rnn_type not in rnn_classes:
            raise ValueError(f"Invalid TextRCNN rnn_type {rnn_type}, expected one of {list(rnn_classes)}")
        rnn_kwargs = {"nonlinearity": "relu"} if rnn_type == "rnn" else {}
        self.rnn = rnn_classes[rnn_type](
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            **rnn_kwargs,
        )
        self.proj = nn.Linear(2 * hid_dim + emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, emb_dim)
        self.emb_dim = emb_dim
        self.sequence_length = sequence_length
        self.history = []

    def predict(self, sequence):
        """Predicts the next embedding from a (B, k, d) sequence of snapshot embeddings."""
        context, _ = self.rnn(sequence)  # (B, k, 2 * hid): left and right contexts
        y = torch.tanh(self.proj(torch.cat([context, sequence], dim=-1)))  # (B, k, hid)
        pooled = y.max(dim=1).values  # (B, hid)
        return self.fc(self.dropout(pooled))  # (B, d)

    def forward(self, batch, edge_index, **kwargs):
        target = getattr(batch, "graph_emb", None)
        if target is None:
            raise ValueError(
                "The `textrcnn` encoder predicts the graph2vec embedding of each snapshot: it "
                "requires `featurization.used_method=graph2vec`."
            )
        target = target.float().view(1, -1)

        if self.history:
            sequence = torch.cat([h.to(target.device) for h in self.history], dim=0)
        else:
            sequence = target.new_zeros((1, self.emb_dim))
        pred = self.predict(sequence.unsqueeze(0))  # (1, d)

        # The actual embedding of the snapshot is the last element of the next sequences
        self.history.append(target.detach())
        if len(self.history) > self.sequence_length:
            self.history.pop(0)

        # The prediction is a graph-level output: edges get it as their src and dst embeddings
        num_edges = edge_index.size(1)
        h_edges = pred.expand(num_edges, -1)
        return {"h": pred, "h_src": h_edges, "h_dst": h_edges}

    def reset_state(self):
        self.history = []
