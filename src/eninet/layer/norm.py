from __future__ import annotations

import torch
from torch import Tensor


class CoorsNorm(torch.nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = torch.nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=1, keepdim=True)
        # norm = torch.where(coors < 1e-6, torch.ones_like(norm), delta)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class MinusDistance(torch.nn.Module):
    def __init__(self, scale_init=1.0, bias_init=0.0):
        super().__init__()
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = torch.nn.Parameter(scale)
        self.scale = torch.max(0.0, self.scale)
        self.bias = torch.zeros(1)
        self.act = torch.nn.Softplus()

    def forward(self, dist):
        return -self.act(self.scale * dist + self.bias)
