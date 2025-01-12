from __future__ import annotations

import os
os.environ["DGLBACKEND"] = "pytorch"

from typing import Tuple, Dict
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from ase import Atoms

import dgl
import torch
import os
import numpy as np

from dgl.data import DGLDataset
from dgl.data.utils import download
from dgl.data.utils import load_graphs, save_graphs

from graph.converter import Molecule2Graph
from data.data_config import DEFAULT_FLOATDTYPE

class MD17DatasetBase(DGLDataset, metaclass=ABCMeta):
    def __init__(
        self,
        target_name: str,
        converter: Molecule2Graph,
        name: str,
        graph_filename: str,
        label_filename: str,
        linegraph_filename: str = None,
        url: str = None
    ) -> None:
        self.target_name = target_name
        self.converter = converter
        self.graph_filename = graph_filename
        self.label_filename = label_filename
        self.linegraph_filename = linegraph_filename
        super().__init__(name=name, url=url)

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    def has_cache(self) -> bool:
        has_graph_cache = os.path.exists(self.graph_filename)
        has_label_cache = os.path.exists(self.label_filename)
        has_linegraph_cache = os.path.exists(self.linegraph_filename) \
            if self.linegraph_filename else True

        return has_graph_cache and has_label_cache and has_linegraph_cache

    def save(self) -> None:
        save_graphs(self.graph_filename, self.graphs)
        torch.save(self.labels, self.label_filename)
        
        if self.linegraph_filename:
            save_graphs(self.linegraph_filename, self.line_graphs)

    def load(self) -> None:
        self.graphs, _ = load_graphs(self.graph_filename)
        self.labels = torch.load(self.label_filename)
        if self.linegraph_filename:
            self.line_graphs, _ = load_graphs(self.linegraph_filename)

    def __getitem__(self, idx: int) -> Tuple:
        if self.linegraph_filename:
            return (
                self.graphs[idx],
                self.line_graphs[idx],
                {'E': self.labels['E'][idx], 'F': self.labels['F'][idx]}
            )
        return (
            self.graphs[idx],
            None,
            {'E': self.labels['E'][idx], 'F': self.labels['F'][idx]}
        )

    def __len__(self) -> int:
        return self.labels['E'].shape[0]

    
class MD17Dataset(MD17DatasetBase):
    def __init__(self, name: str, **kwargs) -> None:
        url = f'http://quantum-machine.org/gdml/data/npz/md17_{name}.npz'
        super().__init__(name=name, url=url, **kwargs)
        
    def download(self) -> None:
        file_path = os.path.join(self.raw_dir, f'md17_{self.target_name}.npz')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def process(self) -> None:
        npz_path = os.path.join(self.raw_dir, f'md17_{self.target_name}.npz')
        data_dict = np.load(npz_path, allow_pickle=True)

        self.E = data_dict['E']
        self.F = data_dict['F']
        self.R = data_dict['R']
        self.Z = data_dict['z']

        self.labels = {
            'E': torch.tensor(self.E, dtype=DEFAULT_FLOATDTYPE),
            'F': torch.tensor(self.F, dtype=DEFAULT_FLOATDTYPE)
        }
        
        graphs = []
        line_graphs = []
        for idx in tqdm(range(len(self.E)), desc="building graphs for MD17 molecule trajectories.. "):
            atoms = Atoms(positions=self.R[idx], numbers=self.Z)
            graph = self.converter.build_graph(atoms)
            graphs.append(graph)
            
            if self.linegraph_filename:
                line_graph = self.converter.build_line_graph(graph)
                line_graphs.append(line_graph)
                
        self.graphs = graphs
        self.line_graphs = line_graphs if self.linegraph_filename else None
        
        assert len(self.graphs) == len(self.labels['E'])
        if self.line_graphs:
            assert len(self.graphs) == len(self.labels['E']) == len(self.line_graphs)
