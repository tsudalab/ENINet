from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class DataConfig:
    split_seed: int
    task: str
    cutoff: float
    train_batch: int
    infer_batch: int
    n_workers: int
    extensive: bool
    file_savedir: str
    label_norm: bool
    train_size: int
    val_size: int
    test_size: Optional[int]
    max_neigh: int


@dataclass
class ModelConfig:
    g_feat_dim: int
    lg_feat_dim: int
    n_rbf: int
    use_linegraph: bool
    g_aggregation: str
    lg_aggregation: str
    n_interactions: int
    loss_type: str
    loss_energy_weight: float = None
    loss_force_weight: float = None
    loss_per_atom_energy: bool = False
    cal_grad: bool = False


@dataclass
class TrainConfig:
    project_name: str
    learning_rate: float
    lr_warmup_steps: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    early_stopping_patience: int
    gradient_clip_val: Optional[float]
    max_epoch: int
    wandb_log: bool
    cuda: int
    ema_scale: float = None
    ema_scale_y: float = None
    ema_scale_dy: float = None


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(
        data=DataConfig(**config_dict["data"]),
        model=ModelConfig(**config_dict["model"]),
        train=TrainConfig(**config_dict["train"]),
    )
