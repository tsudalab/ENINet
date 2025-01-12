from __future__ import annotations

import os

os.environ["DGLBACKEND"] = "pytorch"

import os

import numpy as np
import torch
from ase import Atoms
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, load_graphs, save_graphs
from tqdm import tqdm

from data.data_config import DEFAULT_FLOATDTYPE
from graph.converter import Molecule2Graph


class QM9Dataset(DGLDataset):
    r"""QM9 dataset for graph property prediction (regression)

    This dataset consists of 130,831 molecules with 12 regression targets.
    Nodes correspond to atoms and edges correspond to close atom pairs.

    This dataset differs from :class:`~dgl.data.QM9EdgeDataset` in the following aspects:
        1. Edges in this dataset are purely distance-based.
        2. It only provides atoms' coordinates and atomic numbers as node features
        3. It only provides 12 regression targets.

    Reference:

    - `"Quantum-Machine.org" <http://quantum-machine.org/datasets/>`_,
    - `"Directional Message Passing for Molecular Graphs" <https://arxiv.org/abs/2003.03123>`_

    Statistics:

    - Number of graphs: 130,831
    - Number of regression targets: 12

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | Keys   | Property                         | Description                                                                       | Unit                                        | Property  |
    +========+==================================+===================================================================================+=============================================+===========+
    | mu     | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | alpha  | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | homo   | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         | Intensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | lumo   | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         | Intensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | gap    | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         | Intensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | r2     | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | zpve   | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | U0     | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | U      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | H      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | G      | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    | Cv     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` | Extensive |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+-----------+
    """

    def __init__(
        self,
        target_name: str,
        converter: Molecule2Graph,
        name: str,
        graph_filename: str,
        label_filename: str,
        linegraph_filename: str = None,
        raw_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
        transform: callable = None,
    ) -> None:

        self.target_name = target_name
        self._url = _get_dgl_url("dataset/qm9_eV.npz")
        self.converter = converter
        self.graph_filename = graph_filename
        self.label_filename = label_filename
        self.linegraph_filename = linegraph_filename
        super(QM9Dataset, self).__init__(
            name=name,
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def has_cache(self) -> bool:
        has_cache = os.path.exists(self.graph_filename) and os.path.exists(
            self.label_filename
        )
        if self.linegraph_filename:
            return has_cache and os.path.exists(self.linegraph_filename)
        return has_cache

    def process(self) -> None:
        npz_path = os.path.join(self.raw_dir, "qm9_eV.npz")
        data_dict = np.load(npz_path, allow_pickle=True)

        self.N = data_dict["N"]
        self.R = data_dict["R"]
        self.Z = data_dict["Z"]
        self.labels = torch.tensor(
            data_dict[self.target_name], dtype=DEFAULT_FLOATDTYPE
        )[:, None]
        # (n_data) -> (n_data, 1)

        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        graphs = []
        line_graphs = []

        for idx in tqdm(
            range(len(self.labels)), desc="Building graphs for QM9 molecules"
        ):
            R = self.R[self.N_cumsum[idx] : self.N_cumsum[idx + 1]]
            Z = self.Z[self.N_cumsum[idx] : self.N_cumsum[idx + 1]]

            atoms = Atoms(positions=R, numbers=Z)
            graph = self.converter.build_graph(atoms)
            graphs.append(graph)

            if self.linegraph_filename:
                line_graph = self.converter.build_line_graph(graph)
                line_graphs.append(line_graph)

        self.graphs = graphs
        self.line_graphs = line_graphs if self.linegraph_filename else None

        assert len(self.graphs) == len(self.labels)
        if self.line_graphs:
            assert len(self.graphs) == len(self.labels) == len(self.line_graphs)

    def download(self) -> None:
        """Download raw data."""
        file_path = os.path.join(self.raw_dir, "qm9_eV.npz")
        if not os.path.exists(file_path):
            download(self._url, path=file_path)

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

    def __getitem__(self, idx: int) -> tuple:
        """Get the graph, line graph (if available), and label by index."""
        if self.linegraph_filename:
            return (self.graphs[idx], self.line_graphs[idx], self.labels[idx])
        return (self.graphs[idx], None, self.labels[idx])

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return self.labels.shape[0]
