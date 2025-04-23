from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract


class CustomMLPDecoder(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout):
        super().__init__(in_dim, out_dim, architecture, dropout)

    def forward(self, h):
        h = self.mlp(h)
        return h
