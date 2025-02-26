from provnet_utils import *
from config import *
import torch.nn as nn
from encoders import TGNEncoder, AncestorEncoder
from experiments.uncertainty import activate_dropout_inference


class Model(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            decoder_few_shot: nn.Module,
            device,
            is_running_mc_dropout,
            use_few_shot,
            freeze_encoder,
        ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.device = device
        self.is_running_mc_dropout = is_running_mc_dropout
        
        self.decoder_few_shot = decoder_few_shot
        self.use_few_shot = use_few_shot
        self.few_shot_mode = False
        self.freeze_encoder = freeze_encoder
        
    def embed(self, batch, full_data, inference=False, **kwargs):
        train_mode = not inference
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            res = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=batch.x,
                x_src=batch.x_src,
                x_dst=batch.x_dst,
                original_n_id=batch.original_n_id,
                msg=batch.msg,
                edge_feats=getattr(batch, "edge_feats", None),
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
                edge_types= batch.edge_type,
                node_type=batch.node_type,
                batch=batch,
            )
        h, h_src, h_dst = self.gather_h(batch, res)
        return h, h_src, h_dst
        
    def forward(self, batch, full_data, inference=False, validation=False):
        train_mode = not inference

        with torch.set_grad_enabled(train_mode):
            h, h_src, h_dst = self.embed(batch, full_data, inference=inference)

            # Train mode: loss | Inference mode: scores
            loss_or_scores = None
            
            for objective in self.decoders:
                results = objective(
                    h_src=h_src, # shape (E, d)
                    h_dst=h_dst, # shape (E, d)
                    h=h, # shape (N, d)
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    y_edge=batch.y,
                    inference=inference,
                    node_type=batch.node_type,
                    node_type_src=batch.node_type_src,
                    node_type_dst=batch.node_type_dst,
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
        return self.to(device)

    # override
    def eval(self):
        super().eval()
        
        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def gather_h(self, batch, res):
        h = res["h"]
        h_src = res.get("h_src", None)
        h_dst = res.get("h_dst", None)
        
        if None in [h_src, h_dst]:
            h_src, h_dst = (h[batch.edge_index[0]], h[batch.edge_index[1]]) \
                if isinstance(h, torch.Tensor) else h
        
        return h, h_src, h_dst
    
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

    def reset_state(self):
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
            