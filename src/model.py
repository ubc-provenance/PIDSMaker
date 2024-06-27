from provnet_utils import *
from config import *
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            graph_reindexer,
        ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
        
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference

        # print("batch:")
        # print(batch)
        
        x = torch.cat([batch.src, batch.dst],dim=0).unique()
        # print(x.shape)
        # x_feat = torch.cat([torch.tensor(batch.x_src.tolist()), torch.tensor(batch.x_dst.tolist())],dim=0)
        # print(x_feat.shape)
        x_feat_src = batch.x_src
        x_feat_dst = batch.x_dst
        edge_index = batch.edge_index
        x_feat = (batch.x_src, batch.x_dst)
        
        # with torch.set_grad_enabled(train_mode):
        #     h = self.encoder(
        #         edge_index=edge_index,
        #         t=batch.t,
        #         x=x,
        #         msg=batch.msg,
        #         edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
        #         full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
        #         inference=inference,
        #     )
       
        # print(x)
        with torch.set_grad_enabled(train_mode):
            h_src = self.encoder(
                features= x_feat_src, 
                node_order = x, 
                adjacency_list = torch.transpose(edge_index,0,1),
                edge_order = batch.t, 
                edge_features = batch.edge_feats, 
                edge_types = batch.t,
            )
            h_dst = self.encoder(
                features= x_feat_dst, 
                node_order = x, 
                adjacency_list = torch.transpose(edge_index,0,1),
                edge_order = batch.t, 
                edge_features = batch.edge_feats, 
                edge_types = batch.t,
            )
            # # TGN encoder returns a pair h_src, h_dst whereas other encoders return simply h
            # # Here we simply transform to get a shape (E, d)
            # h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) \
            #     if isinstance(h, torch.Tensor) \
            #     else h
            
            # # In case TGN is not used, x_src and x_dst have shape (N, d) instead
            # # of (E, d). To be iso with TGN to compute the loss, we transform to shape (E, d)
            # if x[0].shape[0] != edge_index.shape[1]:
            #     x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])
            
            if self.use_contrastive_learning:
                involved_nodes = edge_index.flatten()
                self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
                self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Train mode: loss | Inference mode: edge scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)
            
            for decoder in self.decoders:
                loss = decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    x=x_feat,
                    # edge_index=edge_index,
                    # edge_type=batch.edge_type,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                )
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")

                loss_or_scores = loss_or_scores + loss
                
            return loss_or_scores
            
            # loss = self.decoders(
            #     x = h, 
            #     data = x_feat,
            # )
            # loss_or_scores = loss
                
            # return loss_or_scores
