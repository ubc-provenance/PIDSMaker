from provnet_utils import *
from config import *
import torch.nn as nn
from encoders import TGNEncoder
from experiments.uncertainty import activate_dropout_inference
from decoders import NodeTypePrediction


class Model(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            decoder_few_shot: nn.Module,
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            graph_reindexer,
            node_level,
            is_running_mc_dropout,
            use_few_shot,
            freeze_encoder,
        ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        self.device = device
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
            
        self.node_level = node_level
        self.is_running_mc_dropout = is_running_mc_dropout
        
        self.decoder_few_shot = decoder_few_shot
        self.use_few_shot = use_few_shot
        self.few_shot_mode = False
        self.freeze_encoder = freeze_encoder
        
    def embed(self, batch, full_data, inference=False, **kwargs):
        train_mode = not inference
        x = self._reshape_x(batch)
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=getattr(batch, "edge_feats", None),
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
                edge_types= batch.edge_type
            )
        return h
        
    def forward(self, batch, full_data, inference=False, validation=False):
        train_mode = not inference
        x = self._reshape_x(batch)

        with torch.set_grad_enabled(train_mode):
            h = self.embed(batch, full_data, inference=inference)
            h_src, h_dst, h, x = self.reshape_for_task(batch, h, x)
                    
            # Train mode: loss | Inference mode: scores
            loss_or_scores = None
            
            for objective in self.decoders:
                node_type = self.get_node_type_if_needed(batch, objective)
                    
                results = objective(
                    h_src=h_src, # shape (E, d)
                    h_dst=h_dst, # shape (E, d)
                    h=h, # shape (N, d)
                    x=x,
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    y_edge=batch.y,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                    node_type=node_type,
                    validation=validation,
                )
                loss = results["loss"]
                
                if loss_or_scores is None:
                    loss_or_scores = (torch.zeros(1) if train_mode else \
                        torch.zeros(loss.shape[0], dtype=torch.float)).to(batch.edge_index.device)
                
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            return results
    
    def _reshape_x(self, batch):
        if hasattr(batch, "x"):
            x = batch.x
        else:
            x = (batch.x_src, batch.x_dst)
        return x
        
    def get_val_ap(self):
        # If multiple decoders are used, we take the average of the val scores
        return np.mean([d.get_val_score() for d in self.decoders])

    def to_device(self, device):
        if self.device == device:
            return self
        
        for decoder in self.decoders:
            decoder.graph_reindexer.to(device)
        
        if isinstance(self.encoder, TGNEncoder):
            self.encoder.to_device(device)
            
        self.device = device
        self.graph_reindexer.to(device)
        return self.to(device)

    # override
    def eval(self):
        super().eval()
        
        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def get_node_type_if_needed(self, batch, objective):
        node_type = getattr(batch, "node_type", None)
        # Special case when TGN is used with node type pred, the batch is not already reindexed so we reindex
        # only to get node types as shape (N, d)
        if isinstance(objective.objective, NodeTypePrediction) and node_type is None:
            reindexed_batch = self.graph_reindexer.reindex_graph(batch)
            node_type = reindexed_batch.node_type
        return node_type
    
    def reshape_for_task(self, batch, h, x):
        if isinstance(h, tuple):
            h_src, h_dst = h
        else:
            # TGN encoder returns a pair h_src, h_dst whereas other encoders return simply h as shape (N, d)
            # Here we simply transform to get a shape (E, d)
            h_src, h_dst = (h[batch.edge_index[0]], h[batch.edge_index[1]]) \
                if isinstance(h, torch.Tensor) else h
        
        if self.node_level:
            if isinstance(h, tuple):
                h, _, n_id = self.graph_reindexer._reindex_graph(batch.edge_index, h[0], h[1]) # TODO: duplicate with the one in TGN encoder, remove
                batch.original_n_id = n_id
            if isinstance(x, tuple):
                x, _, n_id = self.graph_reindexer._reindex_graph(batch.edge_index, x[0], x[1])
                batch.original_n_id = n_id
            
        return h_src, h_dst, h, x
    
    def to_fine_tuning(self, do: bool):
        if not self.use_few_shot:
            return
        if do and not self.few_shot_mode:
            
            if self.freeze_encoder:
                self.encoder.eval()
                for param in self.encoder.parameters(): # freeze the encoder
                    param.requires_grad = False
            
            # the decoder is replaced by a copy of the decoder_few_shot + the old decoder is saved for later switch
            ssl_decoder = self.decoders # switch the pretext decoder and fine-tuning decoder
            self.decoders = copy.deepcopy(self.decoder_few_shot)
            self.ssl_decoder = ssl_decoder
            self.few_shot_mode = True
        
        if not do and self.few_shot_mode:
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True
            
            # the ssl decoder is set back
            self.decoders = self.ssl_decoder
            self.few_shot_mode = False
