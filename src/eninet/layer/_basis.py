from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from data.data_config import DEFAULT_FLOATDTYPE


class GaussianRBF(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 5.0,
        n_rbf: int = 64,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.n_rbf = n_rbf
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.n_rbf)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class BesselRBF(torch.nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf
        freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs: Tensor) -> Tensor:
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y
