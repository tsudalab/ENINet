from __future__ import annotations

import os
os.environ["DGLBACKEND"] = "pytorch"

from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from ase import Atoms

import torch
from torch import Tensor
import numpy as np

from dgl import DGLGraph
from dgl.data import DGLDataset
from dgl.data.utils import download, _get_dgl_url
from dgl.data.utils import load_graphs, save_graphs

from graph.converter import GraphConverter
from data.data_config import DEFAULT_FLOATDTYPE

class ASEDataset(DGLDataset):
    def __init__(
        self,
        name: str,
        atoms: List[Atoms],
        labels: Union[List[Tensor], Tensor],
        converter: GraphConverter,
        graph_filename: str,
        label_filename: str,
        linegraph_filename: Optional[str] = None,
        ) -> None:
        self.atoms = atoms
        self.labels = torch.stack(labels) if isinstance(labels, list) else labels
        self.labels = self.labels.to(DEFAULT_FLOATDTYPE)
        self.converter = converter
        self.graph_filename = graph_filename
        self.label_filename = label_filename
        self.linegraph_filename = linegraph_filename

        super().__init__(name=name)
        
    def has_cache(self) -> bool:
        has_graph_cache = os.path.exists(self.graph_filename)
        has_label_cache = os.path.exists(self.label_filename)
        has_linegraph_cache = os.path.exists(self.linegraph_filename) \
            if self.linegraph_filename else True

        return has_graph_cache and has_label_cache and has_linegraph_cache
        
    def process(self) -> None:
        graphs = []
        line_graphs = []
        for i in tqdm(range(len(self.atoms)), desc="building graphs for Atoms.. "):
            atoms = self.atoms[i]
            graph = self.converter.build_graph(atoms)
            graphs.append(graph)
            if self.linegraph_filename:
                line_graph = self.converter.build_line_graph(graph)
                line_graphs.append(line_graph)

        self.graphs = graphs
        self.line_graphs = line_graphs
                
        assert len(self.graphs) == len(self.labels)

    def save(self) -> None:
        """Save dgl graphs and labels."""
        save_graphs(self.graph_filename, self.graphs)
        torch.save(self.labels, self.label_filename)
        if self.linegraph_filename:
            save_graphs(self.linegraph_filename, self.line_graphs)

    def load(self) -> None:
        """Load dgl graphs and labels."""
        self.graphs, _ = load_graphs(self.graph_filename)
        self.labels = torch.load(self.label_filename)
        if self.linegraph_filename:
            self.line_graphs, _ = load_graphs(self.linegraph_filename)

    def __getitem__(self, idx: int) -> Tuple[DGLGraph, Optional[DGLGraph], Tensor]:
        if self.linegraph_filename:
            return self.graphs[idx], self.line_graphs[idx], self.labels[idx]
        return self.graphs[idx], None, self.labels[idx]
    
    def __len__(self) -> int:
        return len(self.graphs)
