from __future__ import annotations

import os

os.environ["DGLBACKEND"] = "pytorch"

import os
from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
import torch
from ase import Atoms
from dgl.data import DGLDataset
from dgl.data.utils import download
from tqdm import tqdm

from eninet.data.data_config import DEFAULT_FLOATDTYPE
from eninet.graph.converter import Molecule2Graph

HARTREE_TO_EV = 27.211386246


class ANIDatasetBase(DGLDataset, metaclass=ABCMeta):
    def __init__(
        self,
        converter: Molecule2Graph,
        name: str,
        n_strcuts: int = 2000000,
        url: str = None,
    ):
        self.n_strcuts = n_strcuts
        self.converter = converter
        self.h5reader: h5py.File = None
        self.mode: str = "preprocess"
        self.h5_file: str = ""
        super().__init__(name=name, url=url)

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    def has_cache(self) -> bool:
        return os.path.exists(self.h5_file)

    def save(self) -> None:
        pass

    def load(self) -> None:
        self.h5reader = h5py.File(self.h5_file, "r")

    def __getitem__(self, idx: int):
        label = torch.tensor(
            self.h5reader[str(idx)]["label"][()], dtype=DEFAULT_FLOATDTYPE
        ).view(1, 1)

        if self.mode == "preprocess":
            n_atoms = len(self.h5reader[str(idx)]["node_types"][()])
            mol_idx = self.h5reader[str(idx)]["mol_idx"][()]
            return n_atoms, mol_idx, label

        atoms = Atoms(
            positions=self.h5reader[str(idx)]["pos"][()],
            numbers=self.h5reader[str(idx)]["node_types"][()],
        )
        graph = self.converter.build_graph(atoms)

        return graph, None, label

    def __len__(self) -> int:
        return len(self.h5reader.keys())


class ANIDataset(ANIDatasetBase):
    _ELEMENT_ENERGIES = np.array(
        [0, -0.500607632585, 0, 0, 0, 0, -37.8302333826, -54.5680045287, -75.0362229210]
    )

    def __init__(self, name: str, h5_file: str, **kwargs):
        self.mode = "preprocess"
        self.ani1_url = "https://ndownloader.figshare.com/files/9057631"
        self.h5_file = h5_file
        super().__init__(name=name, url=self.ani1_url, **kwargs)

    @property
    def file_path(self) -> str:
        return f"{self.raw_dir}/{self.name}_release"

    def download(self) -> None:
        if not os.path.exists(self.file_path):
            download(self.ani1_url, path=self.file_path)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def process(self) -> None:
        atomic_numbers = {b"H": 1, b"C": 6, b"N": 7, b"O": 8}

        os.makedirs(os.path.dirname(self.h5_file), exist_ok=True)
        with h5py.File(self.h5_file, "w") as fw:
            mol_idx = total_idx = 0
            for i in tqdm(range(1, 9), desc="Reading h5 files"):
                h5_file = f"ani_gdb_s0{i}.h5"
                with h5py.File(os.path.join(self.file_path, h5_file), "r") as g:
                    molecules = list(g.values())[0].items()
                    for mol_id, mol in tqdm(
                        molecules, desc="Converting molecules", leave=False
                    ):
                        z = np.array([atomic_numbers[atom] for atom in mol["species"]])
                        all_pos = mol["coordinates"]
                        all_y = mol["energies"] * HARTREE_TO_EV

                        assert all_pos.shape[0] == all_y.shape[0]
                        assert all_pos.shape[1] == z.shape[0]
                        assert all_pos.shape[2] == 3

                        for pos, y in zip(all_pos, all_y):
                            group = fw.create_group(f"{total_idx}")
                            group.create_dataset("node_types", data=z)
                            group.create_dataset("pos", data=pos)
                            y = y - np.sum(self._ELEMENT_ENERGIES[z]) * HARTREE_TO_EV
                            group.create_dataset("label", data=y)
                            group.create_dataset("mol_idx", data=mol_idx)
                            total_idx += 1

                        mol_idx += 1

        self.h5reader = h5py.File(self.h5_file, "r")
