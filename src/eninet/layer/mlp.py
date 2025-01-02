from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


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
        self.activation = torch.nn.SiLU if activation == "silu" else None
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers
        layers = []
        layers.append(torch.nn.Linear(n_input, hidden_layers[0]))
        layers.append(self.activation(**activation_kwargs))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation(**activation_kwargs))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))

        if self.activate_final:
            layers.append(self.activation(**activation_kwargs))

        self.net = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                self.w_init(layer.weight.data)
                self.b_init(layer.bias.data)

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
        self.activation = torch.nn.SiLU if activation == "silu" else None
        self.activate_final = activate_final
        self.w_init = torch.nn.init.xavier_uniform_
        self.b_init = torch.nn.init.zeros_

        # Create layers
        layers = []
        gates = []
        layers.append(torch.nn.Linear(n_input, hidden_layers[0]))
        layers.append(self.activation(**activation_kwargs))
        gates.append(torch.nn.Linear(n_input, hidden_layers[0]))
        gates.append(self.activation(**activation_kwargs))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation(**activation_kwargs))
            gates.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            gates.append(self.activation(**activation_kwargs))

        layers.append(torch.nn.Linear(hidden_layers[-1], n_output))
        gates.append(torch.nn.Linear(hidden_layers[-1], n_output))
        gates.append(torch.nn.Sigmoid())

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

    def forward(self, s, v):
        v_W = self.linear_v(v)
        v_U, v_V = torch.chunk(v_W, 2, dim=-1)

        channel_mix = self.mix(torch.cat([s, v_V.norm(dim=1, keepdim=True)], dim=-1))

        out_s, v_gate = torch.chunk(channel_mix, 2, dim=-1)
        out_v = v_U * v_gate

        if self.final_act:
            out_s = self.act(out_s)

        return out_s, out_v
