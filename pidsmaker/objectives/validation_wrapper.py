import torch
import torch.nn as nn

from pidsmaker.utils.utils import log


class ValidationWrapper(nn.Module):
    """
    Binded to a an objective object to store the scores seen in the validation set.
    """

    def __init__(self, objective, graph_reindexer, is_edge_type_prediction, use_few_shot):
        super().__init__()
        self.objective = objective
        self.graph_reindexer = graph_reindexer
        self.is_edge_type_prediction = is_edge_type_prediction

        self.use_few_shot = use_few_shot
        self.losses = []
        self.ys = []

    def get_val_score(self) -> float:
        if self.use_few_shot:
            # Chunks of scores are concatenated once (rather than one 0-d tensor per edge)
            self.losses = torch.cat([l.reshape(-1) for l in self.losses]) if self.losses else torch.tensor([])
            self.ys = torch.cat([y.reshape(-1) for y in self.ys]) if self.ys else torch.tensor([])
            malicious_mean = self.losses[self.ys == 1].mean()
            benign_mean = self.losses[self.ys == 0].mean()
            log(f"=> Total mean score of fake malicious edges: {malicious_mean:.4f}")
            log(f"=> Total mean score of benign malicious edges: {benign_mean:.4f}")
            # ap = ap_score(self.ys, self.losses)
            score = malicious_mean - benign_mean
        elif hasattr(self.objective, "get_val_score"):
            # Objectives measuring their own validation quality (e.g. link prediction AUC + AP)
            score = self.objective.get_val_score()
        else:
            score = 0.0

        self.losses = []
        self.ys = []
        return score

    def forward(self, edge_type, y_edge, validation, inference, *args, **kwargs):
        results = self.objective(
            *args, edge_type=edge_type, y_edge=y_edge, inference=inference, validation=validation, **kwargs
        )
        assert isinstance(results, dict), "Return value of an objective should be a dict"

        loss_or_losses = results["loss"]

        if validation and self.use_few_shot:
            self.losses.append(loss_or_losses.detach().cpu())
            self.ys.append(y_edge.cpu())

        return results
