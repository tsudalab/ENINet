import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from data.collate_fn import collate_fn_g_pes, collate_fn_lg_pes
from data.data_config import DEFAULT_FLOATDTYPE
from data.md17_dataset import MD17Dataset
from data.scaler import StandardScaler
from graph.converter import Molecule2Graph
from model.config import load_config
from model.pl_wrapper import PESModule
from script.train_utils import (
    parse,
    setup_data,
    setup_model,
    setup_trainer,
    setup_wandb,
)

torch.set_default_dtype(DEFAULT_FLOATDTYPE)


def main():
    args = parse()
    config = load_config(args.config)
    print(config)

    wandb_logger = setup_wandb(config)

    # Setup Converter
    converter = Molecule2Graph(cutoff=config.data.cutoff)

    # Data Setup
    train_data, val_data, test_data, scaler = setup_data(
        data_config=config.data,
        converter=converter,
        dataset_class=MD17Dataset,
        dataset_name="md17",
        scaler_class=StandardScaler,
        build_linegraph=False,
    )

    collate_fn = collate_fn_lg_pes if config.model.use_linegraph else collate_fn_g_pes

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
        model_class=PESModule,
        n_elements=len(converter.element_types),
        scaler=scaler,
        cutoff=config.data.cutoff,
        is_extensive=config.data.extensive,
    )

    # Trainer Setup
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=f"task_{config.data.task}",
        filename=f"{config.data.task}" + "-{epoch:02d}-{val_mae:.3f}",
    )

    earlystopping_callback = EarlyStopping(
        "val_loss", mode="min", patience=config.train.early_stopping_patience
    )
    trainer = setup_trainer(
        config,
        wandb_logger,
        checkpoint_callback,
        earlystopping_callback,
        infer_mode=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
