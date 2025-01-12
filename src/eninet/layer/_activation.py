from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ShiftedSoftplus(nn.Module):
    r"""Shifted version of softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


activation_dict = {
    "silu": torch.nn.SiLU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "leaky_relu": torch.nn.LeakyReLU,
    "softplus": torch.nn.Softplus,
    "shifted_softplus": ShiftedSoftplus,
}
