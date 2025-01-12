from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from layer._activation import activation_dict


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "silu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        activate_final: bool = False,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.activation = activation_dict.get(activation, torch.nn.SiLU)
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers
        layers = [
            torch.nn.Linear(n_input, hidden_layers[0]),
            self.activation(**activation_kwargs),
        ]

        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(self.activation(**activation_kwargs))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))

        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))

        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight)
                self.b_init(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GateMLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "silu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        activate_final: bool = False,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.activation = activation_dict.get(activation, torch.nn.SiLU)
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers and gates
        layers = []
        gates = []
        in_features = n_input

        for out_features in hidden_layers:
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(self.activation(**activation_kwargs))
            gates.append(torch.nn.Linear(in_features, out_features))
            gates.append(self.activation(**activation_kwargs))
            in_features = out_features

        layers.append(torch.nn.Linear(in_features, n_output))
        gates.append(torch.nn.Linear(in_features, n_output))
        gates.append(self.activation(**activation_kwargs))

        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))

        self.net = torch.nn.Sequential(*layers)
        self.gates = torch.nn.Sequential(*gates)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)
        for layer in self.gates:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) * self.gates(x)


class GatedEquiBlock(torch.nn.Module):
    def __init__(self, feat_dim: int, out_dim: int, final_act: bool = False):
        super().__init__()

        self.linear_v = nn.Linear(feat_dim, out_dim * 2, bias=False)
        self.mix = MLP(
            n_input=feat_dim + out_dim,
            n_output=out_dim * 2,
            hidden_layers=(feat_dim + out_dim,),
        )

        self.final_act = final_act
        if final_act:
            self.act = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_W = self.linear_v(v)
        v_U, v_V = torch.chunk(v_W, 2, dim=-1)

        v_V_norm = v_V.norm(dim=1, keepdim=True)
        s_v_concat = torch.cat([s, v_V_norm], dim=-1)
        channel_mix = self.mix(s_v_concat)

        out_s, v_gate = torch.chunk(channel_mix, 2, dim=-1)
        out_v = v_U * v_gate

        if self.final_act:
            out_s = self.act(out_s)

        return out_s, out_v
