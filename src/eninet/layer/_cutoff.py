from __future__ import annotations

import torch
from torch import Tensor

from eninet.data.data_config import DEFAULT_FLOATDTYPE


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: float = 5.0) -> None:
        super().__init__()
        self.register_buffer(
            "cutoff", torch.as_tensor(cutoff, dtype=DEFAULT_FLOATDTYPE)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = 0.5 * (1 + torch.cos(torch.pi * x / self.cutoff))
        mask = x <= self.cutoff
        return out * mask


class PolynomialCutoff(torch.nn.Module):
    """
    https://runner.pages.gwdg.de/runner/1.2/theory/symmetryfunctions/
    """

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.register_buffer(
            "cutoff", torch.as_tensor(cutoff, dtype=DEFAULT_FLOATDTYPE)
        )

    def forward(self, x: Tensor):
        x = x / self.cutoff
        out = (x * (x * (x * (315 - 70 * x) - 540) + 420) - 126) * (x**5) + 1
        mask = x <= self.cutoff
        return out * mask
