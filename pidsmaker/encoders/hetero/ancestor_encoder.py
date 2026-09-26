import torch
import torch.nn as nn

from pidsmaker.utils.dataset_utils import rel2id_darpa_tc


class AncestorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, encoder, num_nodes, device):
        super().__init__()
        self.rnn = nn.GLSTM(in_dim + edge_dim, out_dim, batch_first=True)
        self.hidden_states = {}  # {process_id: (h, c)}
        self.embedding_store = torch.zeros((num_nodes, out_dim), device=device)
        self.edge_dim = edge_dim

        considered_events = ["EVENT_EXECUTE", "EVENT_CLONE"]
        self.considered_events = torch.tensor(
            [rel2id_darpa_tc[event] - 1 for event in considered_events], device=device
        )

        self.encoder = encoder
        self.linear = nn.Linear(out_dim + in_dim, out_dim)

    def forward(self, edge_index, *args, **kwargs):
        x = self._forward(edge_index=edge_index, *args, **kwargs)

        src, dst = edge_index
        for arg in ["x_src", "x_dst", "x"]:
            kwargs.pop(arg, None)
        res = self.encoder(edge_index=edge_index, x=x, x_src=x[src], x_dst=x[dst], **kwargs)
        return res

    def _forward(self, edge_index, edge_types, x, original_n_id, **kwargs):
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        mask = torch.isin(edge_types.argmax(dim=1), self.considered_events)
        edge_index, edge_types, x_src, x_dst = (
            edge_index[:, mask],
            edge_types[mask],
            x_src[mask],
            x_dst[mask],
        )  # NOTE: check here if the filtered edge_index makes sense

        batch_inputs = torch.cat([edge_types, x_src], dim=1)
        process_ids = original_n_id[edge_index[1]]
        process_features = x_dst

        if batch_inputs.shape[0] > 0:
            # TODO: multiple forward are needed to handle duplicate destination nodes
            x_rnn = self._process_batch(batch_inputs, process_ids, process_features)
            self.embedding_store[process_ids] = x_rnn.detach()

            embeddings = self.embedding_store[original_n_id]
            mask = embeddings.any(dim=1)
            new_x = x.clone()
            new_x[mask] = self.linear(torch.cat([new_x[mask], embeddings[mask]], dim=1))
            return new_x

        return x

    def _process_batch(self, emb_sequence, process_ids, process_features):
        """Runs an incremental forward pass for a batch of processes."""
        # Get hidden states for each process, initializing if missing
        h_0, c_0 = [], []
        for nid, feats in zip(process_ids, process_features):
            if nid not in self.hidden_states:
                # base_state = torch.cat([feats.reshape(1, 1, -1), torch.zeros((1, 1, self.edge_dim), device=feats.device)], dim=-1)
                # self.hidden_states[nid] = (
                #     base_state, # h_0
                #     base_state, # c_0
                # )
                self.hidden_states[nid] = (
                    torch.zeros(1, 1, self.rnn.hidden_size, device=feats.device),  # h_0
                    torch.zeros(1, 1, self.rnn.hidden_size, device=feats.device),  # c_0
                )
            h, c = self.hidden_states[nid]
            h_0.append(h)
            c_0.append(c)

        # Stack hidden states into batch format
        h_0 = torch.cat(h_0, dim=1)  # (1, batch_size, out_dim)
        c_0 = torch.cat(c_0, dim=1)  # (1, batch_size, out_dim)

        # Run RNN forward pass
        output, (new_h, new_c) = self.rnn(emb_sequence.unsqueeze(1), (h_0, c_0))

        # Update hidden states
        for i, nid in enumerate(process_ids):
            self.hidden_states[nid] = (
                new_h[:, i : i + 1, :].detach(),
                new_c[:, i : i + 1, :].detach(),
            )

        return output[:, -1, :]  # Return last output embedding

    def reset_state(self):
        self.hidden_states = {}
        self.embedding_store = {}
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
