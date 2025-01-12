from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from dgl import DGLGraph
from torch import Tensor
from torch.nn import Module

from data.scaler import BaseScaler
from layer import (
    EquivariantDipoleReadout,
    EquivariantElectronicSpatialExtent,
    EquivariantPolarizabilityReadout,
    EquivariantScalarReadout,
    ScalarReadout,
)
from layer._error import TensorPerAtomRMSE
from layer._utils import EMA
from model.model import EquiThreeBody


class BaseTrainModule(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        lr_warmup_steps: int = 10000,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.6,
        weight_decay: float = 0.0,
        task_type: str = "regression",
        is_extensive: bool = True,
        cal_grad: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.weight_decay = weight_decay
        self.task_type = task_type
        if task_type == "classification":
            raise NotImplementedError()
        self.is_extensive = is_extensive
        self.cal_grad = cal_grad
        self.save_hyperparameters(ignore="kwargs")

    @abstractmethod
    def forward(self, atoms_graph: DGLGraph, line_graph: Optional[DGLGraph]):
        pass

    @abstractmethod
    def compute_loss(self, labels: Tensor, output: Tensor):
        pass

    @abstractmethod
    def compute_metrics(self, labels, output, batch_num_nodes):
        pass

    def step(self, batch, phase):
        graph, line_graph, labels = batch
        output = self(graph, line_graph)
        step_log = {
            f"{phase}_loss": self.compute_loss(
                labels, output, graph.batch_num_nodes(), phase
            )
        }

        if phase == "train":
            step_log["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]

        metrics = self.compute_metrics(labels, output, graph.batch_num_nodes())
        step_log.update({f"{phase}_{key}": value for key, value in metrics.items()})

        return step_log

    def training_step(
        self, batch: Tuple[DGLGraph, Optional[DGLGraph], Tensor], batch_idx: int
    ) -> Tensor:
        step_log = self.step(batch, phase="train")
        self.log_dict(step_log, prog_bar=True, batch_size=batch[0].batch_size)
        return step_log["train_loss"]

    def validation_step(
        self, batch: Tuple[DGLGraph, Optional[DGLGraph], Tensor], batch_idx: int
    ) -> Tensor:
        if self.cal_grad:
            torch.set_grad_enabled(True)
        step_log = self.step(batch, phase="val")
        self.log_dict(
            step_log, prog_bar=True, batch_size=batch[0].batch_size, on_step=True
        )
        return step_log["val_loss"]

    def test_step(
        self, batch: Tuple[DGLGraph, Optional[DGLGraph], Tensor], batch_idx: int
    ) -> Tensor:
        if self.cal_grad:
            torch.set_grad_enabled(True)
        step_log = self.step(batch, phase="test")
        self.log_dict(
            step_log, prog_bar=True, batch_size=batch[0].batch_size, on_step=True
        )
        return step_log["test_loss"]

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience,
            factor=self.reduce_lr_factor,
            min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def optimizer_step(self, *args, **kwargs) -> None:
        optimizer = kwargs.get("optimizer", args[2])
        if self.trainer.global_step < self.lr_warmup_steps:
            lr_scale = min(1.0, (self.trainer.global_step + 1) / self.lr_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()


class ScalarPredModule(BaseTrainModule):
    def __init__(
        self,
        n_elements: int,
        scaler: BaseScaler,
        g_feat_dim: int = 128,
        lg_feat_dim: int = 16,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        use_linegraph: bool = True,
        g_aggregation: str = "sum",
        lg_aggregation: str = "sum",
        ema_scale: float = 0.99,
        loss_type: str = "mae",
        readout_module: str = "EquivariantScalar",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_layers = EquiThreeBody(
            n_elements=n_elements,
            g_feat_dim=g_feat_dim,
            lg_feat_dim=lg_feat_dim,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            g_aggregation=g_aggregation,
            lg_aggreation=lg_aggregation,
            use_linegraph=use_linegraph,
        )
        self.ema = EMA(ema_scale)
        self.readout = self._get_readout_module(readout_module, g_feat_dim)
        self.loss = self._get_loss_function(loss_type)
        self.scaler = scaler

    def _get_readout_module(self, readout_module: str, g_feat_dim: int) -> Module:
        if readout_module == "EquivariantScalar":
            return EquivariantScalarReadout(g_feat_dim)
        elif readout_module == "Scalar":
            return ScalarReadout(g_feat_dim, [g_feat_dim // 2, g_feat_dim // 2])
        elif readout_module == "Dipole":
            return EquivariantDipoleReadout(g_feat_dim)
        elif readout_module == "ElectronicSpatialExtent":
            return EquivariantElectronicSpatialExtent(g_feat_dim)
        else:
            raise ValueError(f"Unknown readout module: {readout_module}")

    def _get_loss_function(self, loss_type: str) -> Callable[[Tensor, Tensor], Tensor]:
        if loss_type == "mae":
            return torch.nn.functional.l1_loss
        elif loss_type == "mse":
            return torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_loss(
        self, labels: Tensor, output: Tensor, batch_num_nodes: Tensor, phase: str
    ) -> Tensor:
        loss = self.loss(output, labels)
        if self.ema.scale < 1.0 and phase != "test":
            loss = self.ema.apply(loss, phase)
        return loss

    def compute_metrics(
        self, labels: Tensor, output: Tensor, batch_num_nodes: Tensor
    ) -> Dict[str, Tensor]:
        mae = torch.nn.functional.l1_loss(output, labels)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(output, labels))
        per_atom_mae = torch.nn.functional.l1_loss(
            labels / batch_num_nodes.view(-1, 1), output / batch_num_nodes.view(-1, 1)
        )
        return {"mae": mae, "rmse": rmse, "per_atom_mae": per_atom_mae}

    def forward(self, atoms_graph: DGLGraph, line_graph: Optional[DGLGraph]) -> Tensor:
        atoms_graph = self.conv_layers(atoms_graph, line_graph)
        atoms_graph = self.readout.pre_reduce(atoms_graph)
        if self.scaler.per_atom:
            atoms_graph.ndata["s"] = self.scaler.inv_transform(atoms_graph.ndata["s"])
            output = self.readout.atom_aggregate(atoms_graph)
        else:
            output = self.readout.mol_aggregate(atoms_graph)
            output = self.scaler.inv_transform(output)
        return output

    def predict_step(
        self,
        batch: Tuple[DGLGraph, Optional[DGLGraph], Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        graph, line_graph, _ = batch
        return self(graph, line_graph)


class TensorPredModule(BaseTrainModule):
    def __init__(
        self,
        n_elements: int,
        scaler: BaseScaler,
        g_feat_dim: int = 128,
        lg_feat_dim: int = 16,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        use_linegraph: bool = True,
        g_aggregation: str = "sum",
        lg_aggregation: str = "sum",
        ema_scale: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_layers = EquiThreeBody(
            n_elements=n_elements,
            g_feat_dim=g_feat_dim,
            lg_feat_dim=lg_feat_dim,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            g_aggregation=g_aggregation,
            lg_aggreation=lg_aggregation,
            use_linegraph=use_linegraph,
        )
        self.ema = EMA(ema_scale)
        self.readout = EquivariantPolarizabilityReadout(g_feat_dim)
        self.loss = torch.nn.functional.mse_loss
        self.scaler = scaler
        self.tensor_rmse = TensorPerAtomRMSE()

    def compute_loss(
        self, labels: Tensor, output: Tensor, batch_num_nodes: Tensor, phase: str
    ) -> Tensor:
        loss = self.loss(output, labels)
        if self.ema.scale < 1.0 and phase != "test":
            loss = self.ema.apply(loss, phase)
        return loss

    def compute_metrics(
        self, labels: Tensor, output: Tensor, batch_num_nodes: Tensor
    ) -> Dict[str, Tensor]:
        batch_num_nodes = batch_num_nodes.view(-1, 1, 1)
        per_atom_rmse_all = self.tensor_rmse(output, labels, batch_num_nodes.squeeze())

        scalar_output = torch.diagonal(output, dim1=1, dim2=2).sum(dim=1) / (3**0.5)
        scalar_labels = torch.diagonal(labels, dim1=1, dim2=2).sum(dim=1) / (3**0.5)
        per_atom_rmse_scalar = self.tensor_rmse(
            scalar_output, scalar_labels, batch_num_nodes.squeeze()
        )

        tensor_output = torch.stack(
            [
                output[:, 0, 1],
                output[:, 1, 2],
                output[:, 0, 2],
                (2 * output[:, 2, 2] - output[:, 0, 0] - output[:, 1, 1])
                / (2 * (3**0.5)),
                (output[:, 0, 0] - output[:, 1, 1]) / 2,
            ],
            dim=1,
        ) * (2**0.5)

        tensor_labels = torch.stack(
            [
                labels[:, 0, 1],
                labels[:, 1, 2],
                labels[:, 0, 2],
                (2 * labels[:, 2, 2] - labels[:, 0, 0] - labels[:, 1, 1])
                / (2 * (3**0.5)),
                (labels[:, 0, 0] - labels[:, 1, 1]) / 2,
            ],
            dim=1,
        ) * (2**0.5)

        per_atom_rmse_tensor = self.tensor_rmse(
            tensor_output, tensor_labels, batch_num_nodes.squeeze()
        )

        assert torch.allclose(
            per_atom_rmse_all**2,
            per_atom_rmse_scalar**2 + per_atom_rmse_tensor**2,
            atol=1e-5,
        )

        return {
            "rmse_all": per_atom_rmse_all,
            "rmse_scalar": per_atom_rmse_scalar,
            "rmse_tensor": per_atom_rmse_tensor,
        }

    def forward(self, atoms_graph: DGLGraph, line_graph: Optional[DGLGraph]) -> Tensor:
        atoms_graph = self.conv_layers(atoms_graph, line_graph)
        atoms_graph = self.readout.pre_reduce(atoms_graph)

        output = self.readout.mol_aggregate(atoms_graph)
        output = self.scaler.inv_transform(output)
        return output

    def test_step(
        self, batch: Tuple[DGLGraph, Optional[DGLGraph], Tensor], batch_idx: int
    ) -> Tensor:
        if self.cal_grad:
            torch.set_grad_enabled(True)
        graph, line_graph, labels = batch
        batch_size = graph.batch_size
        output = self(graph, line_graph)

        loss = self.compute_loss(labels, output, graph.batch_num_nodes(), "test")
        metrics = self.compute_metrics(labels, output, graph.batch_num_nodes())

        test_log = {f"test_{key}": value for key, value in metrics.items()}
        self.log_dict(test_log, prog_bar=True, batch_size=batch_size, on_step=True)
        return loss


class PESModule(BaseTrainModule):
    def __init__(
        self,
        n_elements: int,
        scaler: BaseScaler,
        g_feat_dim: int = 128,
        lg_feat_dim: int = 16,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        use_linegraph: bool = True,
        g_aggregation: str = "sum",
        lg_aggregation: str = "sum",
        ema_scale_y: float = 1.0,
        ema_scale_dy: float = 1.0,
        loss_energy_weight: float = 1.0,
        loss_force_weight: float = 1000.0,
        loss_per_atom_energy: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.conv_layers = EquiThreeBody(
            n_elements=n_elements,
            g_feat_dim=g_feat_dim,
            lg_feat_dim=lg_feat_dim,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            activation=activation,
            g_aggregation=g_aggregation,
            lg_aggreation=lg_aggregation,
            use_linegraph=use_linegraph,
        )
        self.ema_y = EMA(ema_scale_y)
        self.ema_dy = EMA(ema_scale_dy)
        self.readout = EquivariantScalarReadout(g_feat_dim)
        self.scaler = scaler
        self.loss_energy_weight = loss_energy_weight
        self.loss_force_weight = loss_force_weight
        self.loss_per_atom_energy = loss_per_atom_energy

    def compute_loss(
        self,
        labels: Dict[str, Tensor],
        output: Dict[str, Tensor],
        batch_num_nodes: Tensor,
        phase: str,
    ) -> Tensor:
        if self.loss_per_atom_energy:
            batch_num_nodes = batch_num_nodes.view(-1, 1)
            loss_E = torch.nn.functional.mse_loss(
                output["E"] / batch_num_nodes, labels["E"] / batch_num_nodes
            )
        else:
            loss_E = torch.nn.functional.mse_loss(output["E"], labels["E"])

        if self.ema_y.scale < 1.0 and phase != "test":
            loss_E = self.ema_y.apply(loss_E, phase)

        loss_F = torch.nn.functional.mse_loss(output["F"], labels["F"].view(-1, 3))

        if self.ema_dy.scale < 1.0 and phase != "test":
            loss_F = self.ema_dy.apply(loss_F, phase)

        loss = self.loss_energy_weight * loss_E + self.loss_force_weight * loss_F
        return loss

    def compute_metrics(
        self,
        labels: Dict[str, Tensor],
        output: Dict[str, Tensor],
        batch_num_nodes: Tensor,
    ) -> Dict[str, Tensor]:
        batch_num_nodes = batch_num_nodes.view(-1, 1)
        mae_E = torch.nn.functional.l1_loss(output["E"], labels["E"])
        rmse_E = torch.sqrt(torch.nn.functional.mse_loss(output["E"], labels["E"]))
        mae_E_per_atom = torch.nn.functional.l1_loss(output["E"], labels["E"])
        rmse_E_per_atom = torch.sqrt(
            torch.nn.functional.mse_loss(output["E"], labels["E"])
        )
        mae_F = torch.nn.functional.l1_loss(output["F"], labels["F"].view(-1, 3))
        rmse_F = torch.sqrt(
            torch.nn.functional.mse_loss(output["F"], labels["F"].view(-1, 3))
        )
        return {
            "mae_E": mae_E,
            "rmse_E": rmse_E,
            "mae_E_per_atom": mae_E_per_atom,
            "rmse_E_per_atom": rmse_E_per_atom,
            "mae_F": mae_F,
            "rmse_F": rmse_F,
        }

    def forward(
        self, atoms_graph: DGLGraph, line_graph: Optional[DGLGraph]
    ) -> Dict[str, Tensor]:
        if self.cal_grad:
            atoms_graph.ndata["pos"].requires_grad_(True)
            
        atoms_graph = self.conv_layers(atoms_graph, line_graph)
        atoms_graph = self.readout.pre_reduce(atoms_graph)
        atoms_graph.ndata["s"] = self.scaler.inv_transform(atoms_graph.ndata["s"])

        output_E = self.readout.atom_aggregate(atoms_graph)
        pos = atoms_graph.ndata["pos"]
        grad_outputs = torch.ones_like(output_E)

        grad = torch.autograd.grad(
            outputs=output_E,
            inputs=pos,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        output_F = -grad if grad is not None else torch.zeros_like(pos) 

        return {"E": output_E, "F": output_F}
