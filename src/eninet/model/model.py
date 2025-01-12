from __future__ import annotations

import torch
import torch.nn as nn
from dgl import DGLGraph

from layer import (
    CosineCutoff,
    GaussianRBF,
    ThreeBodyEquiGraphConvSimple,
    TwoBodyEquiGraphConv,
)


class EquiThreeBody(torch.nn.Module):

    def __init__(
        self,
        n_elements: int,
        g_feat_dim: int = 128,
        lg_feat_dim: int = 16,
        n_interactions: int = 3,
        n_rbf: int = 20,
        cutoff=5.0,
        activation: str = "silu",
        g_aggregation: str = "sum",
        lg_aggreation: str = "sum",
        use_linegraph: bool = True,
    ) -> None:
        super().__init__()

        self.g_feat_dim = g_feat_dim
        self.lg_feat_dim = lg_feat_dim
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.use_linegraph = use_linegraph

        self.twobody_cutoff_fn = CosineCutoff(cutoff)
        self.threebody_cutoff_fn = CosineCutoff(cutoff * 2)
        self.twobody_dist_rb = GaussianRBF(0, cutoff, n_rbf)
        self.threebody_dist_rb = GaussianRBF(0, cutoff * 2, n_rbf)

        self.n_embedding = nn.Embedding(n_elements, g_feat_dim)
        self.e_embedding = nn.Linear(n_rbf, g_feat_dim)
        self.t_embedding = nn.Linear(n_rbf, lg_feat_dim)

        self.twobody_conv = nn.ModuleList(
            [
                TwoBodyEquiGraphConv(
                    feat_dim=g_feat_dim,
                    n_rbf=n_rbf,
                    activation=activation,
                    aggregation=g_aggregation,
                    cutoff=cutoff,
                )
                for _ in range(n_interactions)
            ]
        )

        if use_linegraph:
            self.threebody_conv = nn.ModuleList(
                [
                    ThreeBodyEquiGraphConvSimple(
                        g_feat_dim=g_feat_dim,
                        lg_feat_dim=lg_feat_dim,
                        n_rbf=n_rbf,
                        activation=activation,
                        aggregation=lg_aggreation,
                        cutoff=cutoff * 2,
                    )
                    for _ in range(n_interactions)
                ]
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.n_embedding.weight)
        nn.init.xavier_uniform_(self.e_embedding.weight)
        nn.init.zeros_(self.e_embedding.bias)
        nn.init.xavier_uniform_(self.t_embedding.weight)
        nn.init.zeros_(self.t_embedding.bias)

    def _compute_neighbor_features(self, g):
        pos_j = g.ndata["pos"][g.edges()[1]]
        if "pbc_offshift" in g.edata:
            pos_j += g.edata["pbc_offshift"]
        pos_i = g.ndata["pos"][g.edges()[0]]
        vctr_ij = pos_j - pos_i
        dist_ij = vctr_ij.norm(dim=1, keepdim=True)

        g.edata["vctr_norm"] = (vctr_ij / dist_ij) * self.cutoff
        g.edata["dist"] = dist_ij
        return g

    def _compute_triplet_features(self, g, l_g):
        pos_k = g.ndata["pos"][g.find_edges(l_g.edges()[1])[1]]
        if "pbc_offshift" in g.edata:
            pos_k += (
                g.edata["pbc_offshift"][l_g.edges()[0]]
                + g.edata["pbc_offshift"][l_g.edges()[1]]
            )
        pos_j = g.ndata["pos"][g.find_edges(l_g.edges()[0])[0]]
        vctr_jk = pos_k - pos_j
        dist_jk = vctr_jk.norm(dim=1, keepdim=True)

        l_g.edata["vctr_norm"] = vctr_jk / dist_jk
        l_g.edata["dist"] = dist_jk
        return l_g

    def _init_atoms(self, g):
        node_s = self.n_embedding(g.ndata["node_type"])[:, None]
        node_v = torch.zeros((node_s.size(0), 3, node_s.size(2)), device=node_s.device)
        return node_s, node_v

    def _init_bonds(self, g):
        dist_ij = g.edata["dist"]
        rb_ij = self.twobody_dist_rb(dist_ij)
        fcut_ij = self.twobody_cutoff_fn(dist_ij)

        edge_s = self.e_embedding(rb_ij) * fcut_ij[..., None]
        edge_v = (
            g.edata["vctr_norm"][..., None].expand(-1, -1, self.g_feat_dim)
            * fcut_ij[..., None]
        )
        return edge_s, edge_v

    def _init_triplets(self, l_g):
        dist_jk = l_g.edata["dist"]
        rb_jk = self.threebody_dist_rb(dist_jk)
        fcut_jk = self.threebody_cutoff_fn(dist_jk)

        triplet_s = self.t_embedding(rb_jk) * fcut_jk[..., None]
        triplet_v = (
            l_g.edata["vctr_norm"][..., None].expand(-1, -1, self.lg_feat_dim)
            * fcut_jk[..., None]
        )
        return triplet_s, triplet_v

    def forward(self, g: DGLGraph, l_g: DGLGraph) -> DGLGraph:
        g = self._compute_neighbor_features(g)
        node_s, node_v = self._init_atoms(g)
        edge_s, edge_v = self._init_bonds(g)

        if self.use_linegraph:
            l_g = self._compute_triplet_features(g, l_g)
            triplet_s, triplet_v = self._init_triplets(l_g)

        for i in range(self.n_interactions):
            if self.use_linegraph:
                edge_s, edge_v, triplet_s, triplet_v = self.threebody_conv[i](
                    l_g, edge_s, edge_v, triplet_s, triplet_v
                )

            node_s, node_v, edge_s, edge_v = self.twobody_conv[i](
                g, node_s, node_v, edge_s, edge_v
            )

        g.ndata["s"] = node_s
        g.ndata["v"] = node_v
        return g
