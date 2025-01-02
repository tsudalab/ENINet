from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class ShiftedSoftplus(torch.nn.Module):
    r"""Shifted version of softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
