import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import CosineCutoff


class EMA:
    def __init__(self, scale: float = 0.999):
        self._scale = scale
        self.ema = {}

    @property
    def scale(self):
        return self._scale

    def apply(self, loss: torch.Tensor, phase: str):
        if phase not in self.ema:
            self.ema[phase] = loss.detach()
        else:
            self.ema[phase] = (
                self._scale * loss + (1 - self._scale) * self.ema[phase].detach()
            )

        return self.ema[phase]
