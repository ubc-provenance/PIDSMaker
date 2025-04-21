import torch.nn as nn


class CustomMLPAsbtract(nn.Module):
    def __init__(self, in_dim, out_dim, architecture, dropout):
        super().__init__()
        self.mlp = build_mlp_from_string(architecture, in_dim, out_dim, dropout)

    @property
    def in_dim(self):
        return self.mlp.in_dim

    @property
    def out_dim(self):
        return self.mlp.out_dim


class CustomModel(nn.Module):
    def __init__(self, in_dim, out_dim, mlp):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        return self.mlp(*args, **kwargs)


def build_mlp_from_string(arch_str, in_dim, out_dim, dropout):
    def parse_layer(layer_str, in_dim):
        layers = []
        parts = layer_str.lower().split(",")

        for part in parts:
            if part.startswith("linear"):
                _, coeff_out = part.split("(")
                coeff_out = float(coeff_out.strip(")"))
                out_dim = int(in_dim * coeff_out)
                layers.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim  # Update in_dim for the next layer

            elif part.startswith("dropout"):
                layers.append(nn.Dropout(dropout))

            elif part == "tanh":
                layers.append(nn.Tanh())

            elif part == "relu":
                layers.append(nn.ReLU())

            elif part == "leaky_relu":
                layers.append(nn.LeakyReLU())

            elif part == "none":
                pass

            else:
                raise ValueError(f"Invalid layer {part}")

        return layers, in_dim

    original_in_dim = in_dim
    arch_str = arch_str.strip().lower().replace(" ", "")
    layer_groups = arch_str.split("|")

    layers = []
    for group in layer_groups:
        group_layers, in_dim = parse_layer(group, in_dim)
        layers.extend(group_layers)
    layers.append(nn.Linear(in_dim, out_dim))

    model = CustomModel(in_dim=original_in_dim, out_dim=out_dim, mlp=nn.Sequential(*layers))
    return model
