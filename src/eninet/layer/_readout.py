from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import ase
import dgl
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import MLP, GatedEquiBlock
from data.data_config import DEFAULT_FLOATDTYPE


class AvgReadout(nn.Module):
    def __init__(self, feat_dim, dims: List[int], field="atom"):
        super().__init__()

        self.mlp = MLP(feat_dim, 1, dims)
        self.field = field

    def forward(self, g: dgl.DGLGraph):
        with g.local_scope():
            if self.field == "atom":
                g.ndata["s"] = g.ndata["s"].squeeze(1)
                output = dgl.mean_nodes(g, "s")
            elif self.field == "bond":
                g.edata["s"] = g.edata["s"].squeeze(1)
                output = dgl.mean_edges(g, "s")
            else:
                raise ValueError(f"field {self.field} not recognized.")
        output = self.mlp(output)
        return output


class ScalarReadout(nn.Module):
    def __init__(self, feat_dim, dims: List[int]):
        super().__init__()

        self.out = MLP(feat_dim, 1, dims)

    def forward(self, g: dgl.DGLGraph):
        g.ndata["s"] = self.out(g.ndata["s"])
        g.ndata["s"] = g.ndata["s"].squeeze(1)

        return g


class EquivariantScalarReadout(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.out = nn.ModuleList(
            [
                GatedEquiBlock(feat_dim, feat_dim // 2, True),
                GatedEquiBlock(feat_dim // 2, 1, False),
            ]
        )

    def pre_reduce(self, g: dgl.DGLGraph):
        # get atom-wise representation
        for o_layer in self.out:
            g.ndata["s"], g.ndata["v"] = o_layer(g.ndata["s"], g.ndata["v"])
        g.ndata["s"] = g.ndata["s"].squeeze(-1)
        g.ndata["v"] = g.ndata["v"].squeeze(-1)
        return g

    def atom_aggregate(self, g: dgl.DGLGraph):
        output = (
            dgl.readout_nodes(g, "s", op="sum")
            + dgl.readout_nodes(g, "v", op="sum").sum(1, keepdim=True)
            * 0  # for gradient
        )
        return output

    def mol_aggregate(self, g: dgl.DGLGraph):
        output = (
            dgl.readout_nodes(g, "s", op="mean")
            + dgl.readout_nodes(g, "v", op="mean").sum(1, keepdim=True) * 0
        )
        return output


class EquivariantDipoleReadout(EquivariantScalarReadout):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim)
        atomic_mass = torch.tensor(ase.data.atomic_masses, dtype=DEFAULT_FLOATDTYPE)
        self.register_buffer("atomic_mass", atomic_mass)

    def atom_aggregate(self, g: dgl.DGLGraph):
        raise NotImplementedError()

    def mol_aggregate(self, g: dgl.DGLGraph):
        with g.local_scope():
            g.ndata["atomic_mass"] = self.atomic_mass[g.ndata["node_type"]].view(-1, 1)
            g.ndata["mass"] = g.ndata["atomic_mass"] * g.ndata["pos"]

            mass_center = dgl.readout_nodes(g, "mass", op="sum") / dgl.readout_nodes(
                g, "atomic_mass", op="sum"
            )
            batch_n_nodes = g.batch_num_nodes()
            indices = torch.arange(g.batch_size, device=g.device).repeat_interleave(
                batch_n_nodes
            )

            g.ndata["dipole"] = g.ndata["v"] + g.ndata["s"] * (
                g.ndata["pos"] - mass_center[indices]
            )
            output = dgl.readout_nodes(g, "dipole", op="sum")

        return output.norm(dim=1, keepdim=True)


class EquivariantDipoleVecReadout(EquivariantScalarReadout):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim)
        atomic_mass = torch.tensor(ase.data.atomic_masses, dtype=DEFAULT_FLOATDTYPE)
        self.register_buffer("atomic_mass", atomic_mass)

    def atom_aggregate(self, g: dgl.DGLGraph):
        raise NotImplementedError()

    def mol_aggregate(self, g: dgl.DGLGraph):
        with g.local_scope():

            g.ndata["atomic_mass"] = self.atomic_mass[g.ndata["node_type"]].view(-1, 1)
            g.ndata["mass"] = g.ndata["atomic_mass"] * g.ndata["pos"]

            mass_center = dgl.readout_nodes(g, "mass", op="sum") / dgl.readout_nodes(
                g, "atomic_mass", op="sum"
            )
            batch_n_nodes = g.batch_num_nodes()
            indices = torch.arange(g.batch_size, device=g.device).repeat_interleave(
                batch_n_nodes
            )

            g.ndata["dipole"] = g.ndata["v"] + g.ndata["s"] * (
                g.ndata["pos"] - mass_center[indices]
            )
            output = dgl.readout_nodes(g, "dipole", op="sum")

        return output


class EquivariantPolarizabilityReadout(EquivariantScalarReadout):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim)
        atomic_mass = torch.tensor(ase.data.atomic_masses, dtype=DEFAULT_FLOATDTYPE)
        self.register_buffer("atomic_mass", atomic_mass)

    def atom_aggregate(self, g: dgl.DGLGraph):
        raise NotImplementedError()

    def mol_aggregate(self, g: dgl.DGLGraph):
        with g.local_scope():
            g.ndata["atomic_mass"] = self.atomic_mass[g.ndata["node_type"]].view(-1, 1)
            g.ndata["mass"] = g.ndata["atomic_mass"] * g.ndata["pos"]

            mass_center = dgl.sum_nodes(g, "mass") / dgl.sum_nodes(g, "atomic_mass")
            batch_n_nodes = g.batch_num_nodes()
            indices = torch.arange(g.batch_size, device=g.device).repeat_interleave(
                batch_n_nodes
            )

            alpha = torch.einsum(
                "bi, bj -> bij",
                g.ndata["v"].squeeze(-1),
                (g.ndata["pos"] - mass_center[indices]),
            )

            g.ndata["alpha"] = (
                torch.eye(3, device=g.device)
                .unsqueeze(0)
                .expand(g.ndata["s"].shape[0], -1, -1)
                * g.ndata["s"].unsqueeze(-1)
                + alpha
                + alpha.transpose(-1, -2)
            )

            output = dgl.sum_nodes(g, "alpha")

        return output


class EquivariantElectronicSpatialExtent(EquivariantScalarReadout):
    def __init__(self, feat_dim: int):
        super().__init__(feat_dim)
        atomic_mass = torch.tensor(ase.data.atomic_masses)
        self.register_buffer("atomic_mass", atomic_mass)

    def atom_aggregate(self, g: dgl.DGLGraph):
        with g.local_scope():
            g.ndata["atomic_mass"] = self.atomic_mass[g.ndata["node_type"]].view(-1, 1)
            g.ndata["mass"] = g.ndata["atomic_mass"] * g.ndata["pos"]
            mass_center = dgl.readout_nodes(g, "mass", op="sum") / dgl.readout_nodes(
                g, "atomic_mass", op="sum"
            )
            batch_n_nodes = g.batch_num_nodes()
            indices = torch.arange(g.batch_size, device=g.device).repeat_interleave(
                batch_n_nodes
            )

            g.ndata["r2"] = g.ndata["v"].sum(dim=1, keepdim=True) * 0 + g.ndata["s"] * (
                (g.ndata["pos"] - mass_center[indices]).norm(dim=1, keepdim=True) ** 2
            )

            output = dgl.readout_nodes(g, "r2", op="sum")

            return output

    def mol_aggregate(self, g: dgl.DGLGraph):
        raise NotImplementedError()
