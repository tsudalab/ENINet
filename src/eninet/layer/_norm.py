from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CoorsNorm(nn.Module):
    """Modified from VecLayerNorm in VisNet"""

    def __init__(
        self, hidden_channels: int, trainable: bool = False, norm_type: str = "max_min"
    ) -> None:
        super(CoorsNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec: Tensor) -> Tensor:
        return vec

    def rms_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist**2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)

        error_tol = 1e-7
        delta = torch.where(delta < error_tol, torch.ones_like(delta), delta)

        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec: Tensor) -> Tensor:
        vec = self.norm(vec)
        return vec * self.weight.unsqueeze(0).unsqueeze(0)
