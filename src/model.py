import torch.nn as nn
from torch import Tensor

from src.config import MLPModelConfig


class MLP(nn.Module):
    def __init__(self, in_dim: int, lin_1_dim: int, lin_2_dim: int, out_dim: int):
        super().__init__()

        relu = nn.ReLU()

        self.linear_1 = nn.Linear(in_dim, lin_1_dim)
        self.relu_1 = relu
        self.linear_2 = nn.Linear(lin_1_dim, lin_2_dim)
        self.relu_2 = relu
        self.linear_3 = nn.Linear(lin_2_dim, out_dim)

        self.layers = nn.Sequential(self.linear_1, self.relu_1, self.linear_2, self.relu_2, self.linear_3)

    def forward(self, data: Tensor) -> Tensor:
        return self.layers(data)


def get_mlp_model(mlp_cfg: MLPModelConfig, in_dim: int, out_dim: int) -> MLP:
    return MLP(
        in_dim=in_dim,
        lin_1_dim=mlp_cfg.linear_1_dim,
        lin_2_dim=mlp_cfg.linear_2_dim,
        out_dim=out_dim,
    )
