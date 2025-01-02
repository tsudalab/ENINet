import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import CosineCutoff


class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        n_elements: int,
        n_rbf: int,
        g_feat_dim: int,
        cutoff: float,
    ):
        super().__init__()
        self.neighbor_embedding = nn.Embedding(n_elements, g_feat_dim)
        self.edge_embedding = nn.Linear(n_rbf, g_feat_dim)
        self.cutoff = CosineCutoff(cutoff)
        self.combine = nn.Linear(g_feat_dim * 2, g_feat_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.neighbor_embedding.weight)
        nn.init.xavier_normal_(self.edge_embedding.weight)
        self.edge_embedding.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.combine.weight)
        self.combine.bias.data.fill_(0.0)

    def _neighbor_proj(self, nodes: dgl.udf.NodeBatch):
        return {
            "x_neighbors": self.neighbor_embedding(nodes.data["node_type"])
        }  # (N_nodes, feat)

    def _edge_proj(self, edges: dgl.udf.EdgeBatch):
        x_edges = self.edge_embedding(
            edges.data["rbf_edges"]
        )  # NOTE: this is not used in torchmd, but used in visnet
        w_edges = self.cutoff(edges.data["dist"])
        return {"x_edges": x_edges * w_edges}

    def _node_combine(self, nodes: dgl.udf.NodeBatch):
        x_combine = torch.cat(
            [nodes.data["x_nodes"], nodes.data["x_integrated"]], dim=-1
        )
        return {"x_combined": self.combine(x_combine)}

    def forward(self, g: dgl.DGLGraph):

        with g.local_scope():
            g.apply_nodes(self._neighbor_proj)
            g.apply_edges(self._edge_proj)
            g.update_all(
                fn.u_mul_e("x_neighbors", "x_edges", "x_integrated"),
                fn.sum("x_integrated", "x_integrated"),
            )

            g.apply_nodes(self._node_combine)

            x_combine = g.ndata["x_combined"]

        return x_combine


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable=False, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist**2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)

    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        # NOTE: test
        # delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        error_tol = 1e-7
        delta = torch.where(delta < error_tol, torch.ones_like(delta), delta)

        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class EMA:
    def __init__(self, scale: float = 0.999):
        self._scale = scale
        self.ema = {"train": None, "val": None}

    @property
    def scale(self):
        return self._scale

    def apply(self, loss: torch.Tensor, phase: str):
        if self.ema[phase] is None:
            self.ema[phase] = loss.detach()

        loss = self._scale * loss + (1 - self._scale) * self.ema[phase]

        self.ema[phase] = loss.detach()

        return loss
