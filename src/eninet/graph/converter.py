from __future__ import annotations

from abc import ABCMeta, abstractmethod

import ase
import dgl
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from data.data_config import DEFAULT_FLOATDTYPE, DEFAULT_INTDTYPE


class GraphConverter(metaclass=ABCMeta):

    @abstractmethod
    def build_graph(self, atoms: Atoms) -> dgl.DGLGraph:
        pass

    def build_line_graph(self, graph: dgl.DGLGraph, **kwargs) -> dgl.DGLGraph:
        raise NotImplementedError()


class Molecule2Graph(GraphConverter):
    def __init__(
        self, zmax: int = 94, cutoff: float = 5.0, max_neighbors: int = 32
    ) -> None:
        self.element_types = ase.data.chemical_symbols[: zmax + 1]
        self.cutoff = cutoff
        self.element_to_index = {
            elem: idx for idx, elem in enumerate(self.element_types)
        }
        self.max_neighbors = max_neighbors

    def build_graph(self, atoms: Atoms) -> dgl.DGLGraph:
        node_type = [
            self.element_to_index[elem] for elem in atoms.get_chemical_symbols()
        ]
        idx_i, idx_j, dist = neighbor_list("ijd", atoms, self.cutoff)

        if self.max_neighbors:
            nonmax_idx = []
            for i in range(len(atoms)):
                i_indices = (idx_i == i).nonzero()[0]
                indices_sorted = np.argsort(dist[i_indices])[: self.max_neighbors]
                nonmax_idx.append(i_indices[indices_sorted])
            nonmax_idx = np.concatenate(nonmax_idx)
            idx_i = idx_i[nonmax_idx]
            idx_j = idx_j[nonmax_idx]

        u, v = torch.tensor(idx_i), torch.tensor(idx_j)
        graph = dgl.graph((u, v), num_nodes=len(atoms))
        graph.ndata["node_type"] = torch.tensor(node_type, dtype=DEFAULT_INTDTYPE)
        graph.ndata["pos"] = torch.tensor(atoms.positions, dtype=DEFAULT_FLOATDTYPE)

        return graph

    def build_line_graph(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        line_graph = graph.line_graph(backtracking=False, shared=False)
        return line_graph
