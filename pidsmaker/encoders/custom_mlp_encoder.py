from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract


class CustomMLPEncoder(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout):
        super().__init__(in_dim, out_dim, architecture, dropout)

    def forward(self, x, **kwargs):
        h = self.mlp(x)
        return {"h": h}
