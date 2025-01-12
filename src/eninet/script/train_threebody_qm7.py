import json
from monty.json import MontyDecoder
import numpy as np
import yaml
import os

import wandb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from dgl.data.utils import split_dataset
from ase import Atoms

from graph.converter import get_element_list, Molecule2Graph
from model.pl_wrapper import TensorPredModule
from data.dataset import ASEDataset
from data.utils import collate_fn_lg, collate_fn_g
from data.scaler import RemoveMeanScaler, StandardScaler, DummyScaler
from data.data_config import DEFAULT_FLOATDTYPE

torch.manual_seed(20240201)
torch.set_default_dtype(DEFAULT_FLOATDTYPE)

import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args

args = parse()

with open(args.config, 'r') as cfg:
    config = yaml.safe_load(cfg)
print(config)

data_config = config.pop('data')
model_config = config.pop('model')
train_config = config.pop('train')

cuda = train_config.pop('cuda')
wandb_log = train_config.pop('wandb_log')
if not wandb_log:
    wandb.init(mode="disabled")
    
task = data_config.pop('task')      # 'CCSD', 'DFT', 'Delta'
task = task.lower()
project_name = f'egnn_qm7_threebody_polarizability_{task}'
wandb_logger = WandbLogger(project=project_name, job_type='train')
wandb_logger.experiment.config.update(dict(data_config))
wandb_logger.experiment.config.update(dict(model_config))
wandb_logger.experiment.config.update(dict(train_config))

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

if task == 'ccsd':
    with open("../../data/CCSD_daDZ_ASE.json", "r") as f:
        ccsd_data = json.load(f, cls=MontyDecoder)
        
    structures = [Atoms.fromdict(d) for d in ccsd_data]
    labels = []
    for struct in structures:
        label = alpha_from_info(struct.info)
        labels.append(torch.tensor(label, dtype=DEFAULT_FLOATDTYPE))

elif task == 'dft':
    with open("../../data/B3LYP_daDZ_ASE.json", "r") as f:
        dft_data = json.load(f, cls=MontyDecoder)
        
    structures = [Atoms.fromdict(d) for d in dft_data]
    labels = []
    for struct in structures:
        label = alpha_from_info(struct.info)
        labels.append(torch.tensor(label, dtype=DEFAULT_FLOATDTYPE))


cutoff = data_config.pop('cutoff')
converter = Molecule2Graph(cutoff=cutoff)

file_savedir = data_config.pop('file_savedir')
dataset = ASEDataset(
    name = "qm7_ccsd_polarizability",
    atoms = structures,
    labels = labels,
    converter = converter,
    graph_filename=os.path.join(file_savedir, f"qm7_graph_cutoff{cutoff}_{task}"),
    label_filename=os.path.join(file_savedir, f"qm7_label_cutoff{cutoff}_{task}"),
    linegraph_filename=os.path.join(file_savedir, f"qm7_linegraph_cutoff{cutoff}_{task}"),
    )

per_atom = data_config.pop('extensive')
scaler = DummyScaler.from_data(
    data=dataset.labels,
    n_atoms=torch.tensor([g.num_nodes() for g in dataset.graphs]).view(-1, 1, 1),
    per_atom=per_atom)

split_seed = data_config.pop('split_seed')
train_ratio = 6000 / len(dataset) 
train_data, val_data, _ = split_dataset(
    dataset,
    frac_list=[train_ratio, 1-train_ratio, 0.],
    shuffle=True,
    random_state=split_seed,
)

batch_size = data_config.pop('batch_size')
n_workers = data_config.pop('n_workers')
if model_config["use_linegraph"]:
    collate_fn = collate_fn_lg
else:
    collate_fn = collate_fn_g
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_workers)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

earlystopping_patience = train_config.pop("early_stopping_patience")
max_epoch = train_config.pop("max_epoch")
gradient_clip_val = train_config.pop("gradient_clip_val")

model = TensorPredModule(
    n_elements=len(converter.element_types),
    scaler=scaler,
    cutoff=cutoff,
    is_extensive=per_atom,
    delta_learning=(task=="delta"),
    **model_config,
    **train_config)

use_line = model_config.pop('use_linegraph')
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath=f"outputs/painn/qm7/polarizability/split{split_seed}/line{use_line}/cutoff{cutoff}",
    filename="eps-{epoch:02d}-{val_loss:.2f}",
)


earlystopping_callback = EarlyStopping(
    "val_loss", mode="min", 
    patience=earlystopping_patience)

trainer = pl.Trainer(
    max_epochs=max_epoch, 
    accelerator="gpu", 
    logger=wandb_logger,
    devices=[cuda], 
    callbacks=[checkpoint_callback, earlystopping_callback],
    gradient_clip_val=gradient_clip_val
    )

trainer.fit(
    model=model, 
    train_dataloaders=train_loader,
    val_dataloaders=val_loader)



# for testing

if task == 'ccsd':
    with open("../../data/CCSD_daDZ_showcase_ASE.json", "r") as f:
        ccsd_data_test = json.load(f, cls=MontyDecoder)
        
    structures_test = [Atoms.fromdict(d) for d in ccsd_data_test]
    labels_test = []
    for struct in structures_test:
        label = alpha_from_info(struct.info)
        labels_test.append(torch.tensor(label, dtype=DEFAULT_FLOATDTYPE))

elif task == 'dft':
    with open("../../data/B3LYP_daDZ_showcase_ASE.json", "r") as f:
        dft_data_test = json.load(f, cls=MontyDecoder)
        
    structures_test = [Atoms.fromdict(d) for d in dft_data_test]
    labels_test = []
    for struct in structures_test:
        label = alpha_from_info(struct.info)
        labels_test.append(torch.tensor(label, dtype=DEFAULT_FLOATDTYPE))
        
elif task == 'delta':
    with open("../../data/B3LYP_daDZ_showcase_ASE.json", "r") as f:
        dft_data = json.load(f, cls=MontyDecoder)
    structures_test = [Atoms.fromdict(d) for d in dft_data]
    
    with open("../../data/CCSD_daDZ_showcase_ASE.json", "r") as f:
        ccsd_data = json.load(f, cls=MontyDecoder)
    structures_ccsd_test = [Atoms.fromdict(d) for d in ccsd_data]
    
    labels_test = []
    for struct_dft, struct_ccsd in zip(structures_test, structures_ccsd_test):
        label_dft = alpha_from_info(struct_dft.info)
        label_ccsd = alpha_from_info(struct_ccsd.info)
        labels_test.append(torch.tensor(np.stack([label_ccsd, label_dft], axis=0), dtype=DEFAULT_FLOATDTYPE))

else:
    raise ValueError(f'Task ({task}) not supported!')

test_data = ASEDataset(
    name = "qm7_ccsd_polarizability_showcase",
    atoms = structures_test,
    labels = labels_test,
    converter = converter,
    graph_filename=os.path.join(file_savedir, f"qm7_showcase_graph_cutoff{cutoff}_{task}"),
    label_filename=os.path.join(file_savedir, f"qm7_showcase_label_cutoff{cutoff}_{task}"),
    linegraph_filename=os.path.join(file_savedir, f"qm7_showcase_linegraph_cutoff{cutoff}_{task}"),
    )

test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn_lg, num_workers=n_workers)

trainer.test(model, dataloaders=test_loader, ckpt_path='best')