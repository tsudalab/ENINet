import argparse
import json
import os
from typing import List, Tuple, Union
import wandb
from ase import Atoms
from monty.json import MontyDecoder

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dgl.data import DGLDataset
from dgl.data.utils import split_dataset

from eninet.data.scaler import BaseScaler
from eninet.graph.converter import GraphConverter
from eninet.model.config import Config, DataConfig
from eninet.model.pl_wrapper import BaseTrainModule



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--ckpt", type=str, help="Path to the checkpoint file", default=None
    )
    parser.add_argument(
        "--infer", type=str, help="Path to the inference dataset", default=None
    )
    args = parser.parse_args()
    return args


def setup_wandb(config: Config):
    if not config.train.wandb_log:
        wandb.init(mode="disabled")

    project_name = config.train.project_name
    wandb_logger = WandbLogger(project=project_name, job_type="train")
    wandb_logger.experiment.config.update(config)
    return wandb_logger


def setup_data(
    data_config: DataConfig,
    converter: GraphConverter,
    dataset_class: DGLDataset,
    dataset_name: str,
    scaler_class: BaseScaler,
    build_linegraph: bool = True,
    structures: list = None,
    labels: list = None,
    infer_mode: bool = False,
) -> Tuple[DGLDataset, DGLDataset, DGLDataset, BaseScaler]:
    # if infer_mode, return only the scaler for inference

    if dataset_class.__name__ in ["QM9Dataset", "MD17Dataset"]:
        dataset_init_args = dict(
            target_name=data_config.task,
            converter=converter,
            name=dataset_name,
            graph_filename=os.path.join(
                data_config.file_savedir,
                f"{dataset_name}_graph_cutoff{data_config.cutoff}_{data_config.task}",
            ),
            label_filename=os.path.join(
                data_config.file_savedir,
                f"{dataset_name}_label_cutoff{data_config.cutoff}_{data_config.task}",
            ),
            linegraph_filename=(
                os.path.join(
                    data_config.file_savedir,
                    f"{dataset_name}_linegraph_cutoff{data_config.cutoff}_{data_config.task}",
                )
                if build_linegraph
                else None
            ),
        )
    elif dataset_class.__name__ in ["CustomDatase"]:
        dataset_init_args = dict(
            name=dataset_name,
            atoms=structures,
            labels=labels,
            converter=converter,
            graph_filename=os.path.join(
                data_config.file_savedir,
                f"{dataset_name}_graph_cutoff{data_config.cutoff}_{data_config.task}",
            ),
            label_filename=os.path.join(
                data_config.file_savedir,
                f"{dataset_name}_label_cutoff{data_config.cutoff}_{data_config.task}",
            ),
            linegraph_filename=(
                os.path.join(
                    data_config.file_savedir,
                    f"{dataset_name}_linegraph_cutoff{data_config.cutoff}_{data_config.task}",
                )
                if build_linegraph
                else None
            ),
        )

    dataset = dataset_class(**dataset_init_args)

    scaler = scaler_class.from_data(
        data=(
            dataset.labels.squeeze()
            if isinstance(dataset.labels, torch.Tensor)
            else dataset.labels["E"]
        ),
        n_atoms=torch.tensor([g.num_nodes() for g in dataset.graphs]),
        per_atom=data_config.extensive,
    )

    if not infer_mode:
        train_ratio = (
            data_config.train_size
            if isinstance(data_config.train_size, float)
            else data_config.train_size / len(dataset)
        )
        val_ratio = (
            data_config.val_size
            if isinstance(data_config.val_size, float)
            else data_config.val_size / len(dataset)
        )

        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[train_ratio, val_ratio, 1 - train_ratio - val_ratio],
            shuffle=True,
            random_state=data_config.split_seed,
        )

        return train_data, val_data, test_data, scaler

    return scaler


def setup_model(
    config: Config,
    model_class: BaseTrainModule,
    n_elements: int,
    scaler: BaseScaler,
    cutoff: float,
    is_extensive: bool,
) -> BaseTrainModule:
    """Sets up the model with its configuration."""
    model = model_class(
        n_elements=n_elements,
        scaler=scaler,
        cutoff=cutoff,
        is_extensive=is_extensive,
        **config.model.__dict__,
        **config.train.__dict__,
    )
    return model


def setup_trainer(
    config: Config,
    wandb_logger,
    checkpoint_callback: ModelCheckpoint,
    earlystopping_callback: EarlyStopping,
    infer_mode: bool = True,
):
    """Sets up the PyTorch Lightning trainer."""

    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        accelerator="gpu",
        logger=wandb_logger,
        devices=[config.train.cuda],
        callbacks=[checkpoint_callback, earlystopping_callback],
        gradient_clip_val=config.train.gradient_clip_val,
        inference_mode=infer_mode,
    )
    return trainer


def read_dataset_from_json(
    json_filename: str, infer_mode: bool = False
) -> Union[Tuple[List, List], List]:
    with open(json_filename, "r") as f:
        dataset = json.load(f, cls=MontyDecoder)

    structures = []
    labels = []
    mol_ids = []

    for mol_id, data in dataset.items():
        structure = Atoms.fromdict(data["structure"])
        structures.append(structure)
        mol_ids.append(mol_id)

        if infer_mode:
            continue

        label = data["label"]
        labels.append([label])

    if infer_mode:
        return mol_ids, structures

    return mol_ids, structures, labels


def print_logo():
    logo = """
   _____  _______  __    __ 
  / __/ |/ /  _/ |/ /__ / /_
 / _//    // //    / -_) __/
/___/_/|_/___/_/|_/\__/\__/ 
                            
    """
    print(logo)
