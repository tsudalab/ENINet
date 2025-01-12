from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch import Tensor


class BaseScaler(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_data(
        cls, data: Tensor, n_atoms: Union[int, Tensor] = 1, per_atom: bool = False
    ) -> "BaseScaler":
        pass


class DummyScaler(BaseScaler):
    """Do nothing about transformation."""

    def __init__(self, per_atom: bool):
        self.per_atom = per_atom

    @classmethod
    def from_data(
        cls, data: Tensor, n_atoms: Union[int, Tensor] = 1, per_atom: bool = False
    ) -> "DummyScaler":
        return cls(per_atom)

    def inv_transform(self, x: Tensor) -> Tensor:
        return x


class StandardScaler(BaseScaler):
    def __init__(self, mean: Tensor, std: Tensor, per_atom: bool = False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.per_atom = per_atom

    @classmethod
    def from_data(
        cls, data: Tensor, n_atoms: Union[int, Tensor] = 1, per_atom: bool = False
    ) -> "StandardScaler":
        if not per_atom:
            n_atoms = 1
        if isinstance(n_atoms, int):
            n_atoms = torch.tensor(
                [n_atoms] * data.size(0), dtype=data.dtype, device=data.device
            ).view(*data.shape)
        elif isinstance(n_atoms, Tensor):
            n_atoms = n_atoms.view(*data.shape)

        mean = torch.mean(data.detach() / n_atoms, dim=0, keepdim=True)
        std = torch.std(data.detach() / n_atoms, dim=0, keepdim=True)

        return cls(mean, std, per_atom)

    def inv_transform(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class RemoveMeanScaler(BaseScaler):
    def __init__(self, mean: Tensor, per_atom: bool = False):
        self.mean = mean
        self.per_atom = per_atom

    @classmethod
    def from_data(
        cls, data: Tensor, n_atoms: Union[int, Tensor] = 1, per_atom: bool = False
    ) -> "RemoveMeanScaler":
        if not per_atom:
            n_atoms = 1
        if isinstance(n_atoms, int):
            n_atoms = torch.tensor(
                [n_atoms] * data.size(0), dtype=data.dtype, device=data.device
            ).view(*data.shape)
        elif isinstance(n_atoms, Tensor):
            n_atoms = n_atoms.view(*data.shape)

        mean = torch.mean(data / n_atoms, dim=0, keepdim=True)
        return cls(mean, per_atom)

    def inv_transform(self, x: Tensor) -> Tensor:
        return x + self.mean.to(x.device)
