import os

import pytorch_lightning as pl
import torch
import wandb
from dgl.data.utils import split_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.data_config import DEFAULT_FLOATDTYPE
from data.qm9_dataset import QM9Dataset
from data.scaler import StandardScaler
from data.utils import collate_fn_g, collate_fn_lg
from graph.converter import GraphConverter, Molecule2Graph
from model.config import DataConfig
from model.pl_wrapper import ScalarPredModule

torch.set_default_dtype(DEFAULT_FLOATDTYPE)

import argparse

from model.config import Config, load_config


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    return args


def setup_wandb(config: Config):
    """Sets up Weights & Biases logging based on config."""
    if not config.train.wandb_log:
        wandb.init(mode="disabled")
        return None

    project_name = f"qm9_{config.data.task}"
    wandb_logger = WandbLogger(project=project_name, job_type="train")
    wandb_logger.experiment.config.update(config)  # Log the entire config
    return wandb_logger


def setup_data(data_config: DataConfig, converter: GraphConverter):
    """Sets up the dataset, scaler, and data loaders."""
    dataset = QM9Dataset(
        label_key=data_config.task,
        converter=converter,
        name="qm9",
        graph_filename=os.path.join(
            data_config.file_savedir,
            f"qm9_graph_cutoff{data_config.cutoff}_{data_config.task}_nmax{data_config.max_neigh}",
        ),
        label_filename=os.path.join(
            data_config.file_savedir,
            f"qm9_label_cutoff{data_config.cutoff}_{data_config.task}_nmax{data_config.max_neigh}",
        ),
        linegraph_filename=os.path.join(
            data_config.file_savedir,
            f"qm9_linegraph_cutoff{data_config.cutoff}_{data_config.task}_nmax{data_config.max_neigh}",
        ),
    )

    scaler = StandardScaler.from_data(
        data=dataset.labels.squeeze(),
        n_atoms=torch.tensor([g.num_nodes() for g in dataset.graphs]),
        per_atom=data_config.extensive,
    )

    if isinstance(data_config.train_size, float):
        train_ratio = data_config.train_size
    else:
        train_ratio = data_config.train_size / len(dataset)
    if isinstance(data_config.val_size, float):
        val_ratio = data_config.val_size
    else:
        val_ratio = data_config.val_size / len(dataset)

    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[train_ratio, val_ratio, 1 - train_ratio - val_ratio],
        shuffle=True,
        random_state=data_config.split_seed,
    )

    return train_data, val_data, test_data, scaler


def setup_model(config: Config, n_elements, scaler, cutoff, is_extensive):
    """Sets up the model with its configuration."""
    model = ScalarPredModule(
        n_elements=n_elements,
        scaler=scaler,
        cutoff=cutoff,
        is_extensive=is_extensive,
        **config.model.__dict__,
        **config.train.__dict__,
    )
    return model


def setup_trainer(config: Config, wandb_logger):
    """Sets up the PyTorch Lightning trainer."""
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_mae",
        mode="min",
        dirpath=f"qm9/task_{config.data.task}",
        filename="eps-{epoch:02d}-{val_mae:.3f}",
    )

    earlystopping_callback = EarlyStopping(
        "val_mae", mode="min", patience=config.train.early_stopping_patience
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        accelerator="gpu",
        logger=wandb_logger,
        devices=[config.train.cuda],
        callbacks=[checkpoint_callback, earlystopping_callback],
        gradient_clip_val=config.train.gradient_clip_val,
    )
    return trainer


def main():
    args = parse()
    config = load_config(args.config)
    print(config)

    wandb_logger = setup_wandb(config)

    # Setup Converter
    converter = Molecule2Graph(cutoff=config.data.cutoff)

    # Data Setup
    train_data, val_data, test_data, scaler = setup_data(config.data, converter)

    if config.model.use_linegraph:
        collate_fn = collate_fn_lg
    else:
        collate_fn = collate_fn_g

    train_loader = DataLoader(
        train_data,
        batch_size=config.data.train_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.n_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.data.train_batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.n_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.data.infer_batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.n_workers,
    )

    # Model Setup
    model = setup_model(
        config,
        n_elements=len(converter.element_types),
        scaler=scaler,
        cutoff=config.data.cutoff,
        is_extensive=config.data.extensive,
    )

    # Trainer Setup
    trainer = setup_trainer(config, wandb_logger)

    # Training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Testing
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
