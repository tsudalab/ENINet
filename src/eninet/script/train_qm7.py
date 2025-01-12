import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import numpy as np

from ase import Atoms
import json
from monty.json import MontyDecoder

from data.data_config import DEFAULT_FLOATDTYPE
from data.dataset import ASEDataset
from data.scaler import DummyScaler
from data.collate_fn import collate_fn_g, collate_fn_lg
from graph.converter import Molecule2Graph
from model.config import load_config
from model.pl_wrapper import TensorPredModule
from script.train_utils import parse, setup_wandb, setup_data, setup_model, setup_trainer

torch.set_default_dtype(DEFAULT_FLOATDTYPE)

def alpha_from_info(struct_info):
    info = list(struct_info)[0].split(",")
    label = np.zeros((3, 3))
    label[0, 0] = float(info[3])
    label[1, 1] = float(info[4])
    label[2, 2] = float(info[5])
    label[0, 1], label[1, 0] = float(info[6]), float(info[6])
    label[0, 2], label[2, 0] = float(info[7]), float(info[7])
    label[1, 2], label[2, 1] = float(info[8]), float(info[8])
    return label


def main():
    args = parse()
    config = load_config(args.config)
    print(config)

    wandb_logger = setup_wandb(config)

    # Setup Converter
    converter = Molecule2Graph(cutoff=config.data.cutoff)
    
    if config.data.task == 'ccsd':
        with open("../data/CCSD_daDZ_ASE.json", "r") as f:
            ccsd_data = json.load(f, cls=MontyDecoder)
        structures = [Atoms.fromdict(d) for d in ccsd_data]
        labels = [torch.tensor(alpha_from_info(struct.info), dtype=DEFAULT_FLOATDTYPE) for struct in structures]
    elif config.data.task == 'dft':
        with open("../data/B3LYP_daDZ_ASE.json", "r") as f:
            dft_data = json.load(f, cls=MontyDecoder)
        structures = [Atoms.fromdict(d) for d in dft_data]
        labels = [torch.tensor(alpha_from_info(struct.info), dtype=DEFAULT_FLOATDTYPE) for struct in structures]
    else:
        raise ValueError(f'Task ({config.data.task}) not supported!')

    
    # Data Setup
    train_data, val_data, test_data, scaler = setup_data(
        data_config=config.data,
        converter=converter,
        dataset_class=ASEDataset,
        dataset_name="qm7",
        scaler_class=DummyScaler,
        structures=structures,
        labels=labels)

    collate_fn = collate_fn_lg if config.model.use_linegraph else collate_fn_g

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
        model_class=TensorPredModule,
        n_elements=len(converter.element_types),
        scaler=scaler,
        cutoff=config.data.cutoff,
        is_extensive=config.data.extensive,
    )

    # Trainer Setup
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_rmse_all",
        mode="min",
        dirpath=f"task_{config.data.task}",
        filename=f"{config.data.task}"+"-{epoch:02d}-{val_rmse_all:.3f}",
    )

    earlystopping_callback = EarlyStopping(
        "val_rmse_all", mode="min", patience=config.train.early_stopping_patience
    )
    trainer = setup_trainer(
        config, 
        wandb_logger, 
        checkpoint_callback, 
        earlystopping_callback, 
        infer_mode=True)

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
