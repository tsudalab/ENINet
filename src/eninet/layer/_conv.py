from __future__ import annotations

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl import DGLGraph

from eninet.data.data_config import DEFAULT_FLOATDTYPE
from eninet.layer._cutoff import CosineCutoff
from eninet.layer._mlp import MLP, GateMLP
from eninet.layer._norm import CoorsNorm

torch.set_default_dtype(DEFAULT_FLOATDTYPE)


class TwoBodyEquiGraphConv(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_rbf: int,
        activation: str,
        aggregation: str = "mean",
        dropout: float = 0.3,
        cutoff: float = 5.0,
    ):

        super(TwoBodyEquiGraphConv, self).__init__()
        self.feat_dim = feat_dim

        self.node_neighbors = nn.Linear(feat_dim * 2, feat_dim)
        self.edge_proj = nn.Linear(feat_dim, feat_dim)

        self.ns_mess = GateMLP(feat_dim, feat_dim, (feat_dim,))

        self.cutoff_fn = CosineCutoff(cutoff)

        self.edge_vec = nn.Linear(feat_dim, feat_dim * 3)

        self.nv_out = torch.nn.Linear(feat_dim, feat_dim * 3, bias=False)
        self.nv_channel = torch.nn.Linear(feat_dim * 2, feat_dim)

        self.nv_proj = torch.nn.Linear(feat_dim, feat_dim * 2, bias=False)
        self.ns_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim * 2), nn.SiLU())

        self.norm_mess = nn.LayerNorm(feat_dim)
        self.norm_nv = CoorsNorm(feat_dim)

        self.aggregation = aggregation

        self.reset_parameters()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def reset_parameters(self):
        layers = [
            self.node_neighbors,
            self.edge_proj,
            self.edge_vec,
            self.nv_channel,
            self.nv_proj,
            self.nv_out,
        ]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)
        self.ns_proj.apply(self.init_weights)
        self.norm_mess.reset_parameters()

    def _update_edge_s(self, edges: dgl.udf.EdgeBatch):
        # fuse node-edge features
        node_neighbors = self.node_neighbors(
            torch.cat([edges.src["n_s"], edges.dst["n_s"]], dim=-1)
        )
        edge_message = node_neighbors * self.edge_proj(edges.data["e_s"])

        es_update = (
            self.ns_mess(edge_message) * self.cutoff_fn(edges.data["dist"])[..., None]
        )

        return {"e_s_upd": es_update}

    def _update_edge_v(self, edges: dgl.udf.EdgeBatch):
        vec_channels = self.edge_vec(edges.data["e_s_upd"])
        node_channel, edge_channel, rel_channel = torch.split(
            vec_channels, self.feat_dim, dim=-1
        )

        e_v_update = (
            edges.src["n_v"] * node_channel
            + edges.data["e_v"] * edge_channel
            + edges.data["vctr_norm"][..., None] * rel_channel
        )
        e_v_update *= self.cutoff_fn(edges.data["dist"])[..., None]

        return {"e_v_upd": e_v_update}

    def _update_node_v(self, nodes: dgl.udf.NodeBatch):
        nv_agg = nodes.data.pop("n_ev")

        nv_out = self.nv_out(nv_agg)
        nv_o1, nv_o2, nv_o3 = torch.chunk(nv_out, 3, dim=-1)

        n_es = nodes.data["n_es"]
        nv_o3_norm = torch.norm(nv_o3, dim=1, keepdim=True)

        v_channel_input = torch.cat([n_es, nv_o3_norm], dim=-1)
        v_channel = self.nv_channel(v_channel_input)

        n_v_update = nv_o1 * v_channel + nv_o2

        return {"n_v_upd": n_v_update}

    def _update_node_s(self, nodes: dgl.udf.NodeBatch):
        n_es = nodes.data.pop("n_es")

        nv_proj = self.nv_proj(nodes.data["n_v_upd"])
        ns_proj = self.ns_proj(n_es)

        nv_o1, nv_o2 = nv_proj.chunk(2, dim=-1)
        ns_o1, ns_o2 = ns_proj.chunk(2, dim=-1)

        nv_dot = torch.sum(nv_o1 * nv_o2, dim=1, keepdim=True)
        n_s_update = nv_dot * ns_o1 + ns_o2

        return {"n_s_upd": n_s_update}

    def forward(
        self,
        g: DGLGraph,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ):
        with g.local_scope():
            g.ndata.update({"n_s": node_s, "n_v": node_v})
            g.edata.update({"e_s": edge_s, "e_v": edge_v})

            g.apply_edges(self._update_edge_s)
            g.apply_edges(self._update_edge_v)

            agg_fn = fn.mean if self.aggregation == "mean" else fn.sum
            g.update_all(fn.copy_e("e_v_upd", "e_v_upd"), agg_fn("e_v_upd", "n_ev"))
            g.update_all(fn.copy_e("e_s_upd", "e_s_upd"), agg_fn("e_s_upd", "n_es"))

            g.apply_nodes(self._update_node_v)
            g.apply_nodes(self._update_node_s)

            node_s = self.norm_mess(g.ndata.pop("n_s_upd") + node_s)
            node_v = self.norm_nv(g.ndata.pop("n_v_upd") + node_v)
            edge_s = g.edata.pop("e_s_upd") + edge_s
            edge_v = g.edata.pop("e_v_upd") + edge_v

        return node_s, node_v, edge_s, edge_v


class ThreeBodyEquiGraphConvSimple(torch.nn.Module):
    def __init__(
        self,
        n_rbf: int,
        activation: str,
        g_feat_dim: int = 128,
        lg_feat_dim: int = 16,
        aggregation: str = "mean",
        cutoff: float = 10.0,
        dropout: float = 0.3,
    ):

        super().__init__()

        self.cutoff_fn = CosineCutoff(cutoff)
        self.g_feat_dim = g_feat_dim
        self.lg_feat_dim = lg_feat_dim

        self.ns_in = nn.Sequential(nn.Linear(g_feat_dim, lg_feat_dim), nn.SiLU())
        self.nv_in = nn.Linear(g_feat_dim, lg_feat_dim, bias=False)

        self.edge_neighbors = nn.Linear(lg_feat_dim * 2, lg_feat_dim)
        self.triplet_proj = nn.Linear(lg_feat_dim, lg_feat_dim)
        self.triplet_gate = GateMLP(lg_feat_dim, lg_feat_dim, (lg_feat_dim,))

        self.norm_mess = nn.LayerNorm(g_feat_dim)
        self.norm_nv = CoorsNorm(g_feat_dim)

        self.triplet_vec = nn.Linear(lg_feat_dim, lg_feat_dim * 3)

        self.out_nv = nn.Linear(lg_feat_dim, g_feat_dim, bias=False)
        self.out_ns = MLP(lg_feat_dim * 2, g_feat_dim, (lg_feat_dim,))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.0)

    def reset_parameters(self):
        layers = [
            self.nv_in,
            self.edge_neighbors,
            self.triplet_proj,
            self.triplet_vec,
            self.out_nv,
        ]
        self.ns_in.apply(self.init_weights)
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.fill_(0.0)
        self.norm_mess.reset_parameters()

    def _update_edge_s(self, edges: dgl.udf.EdgeBatch):
        node_cat = torch.cat([edges.src["n_s"], edges.dst["n_s"]], dim=-1)
        edge_neighbors = self.edge_neighbors(node_cat)
        triplet_message = edge_neighbors * self.triplet_proj(edges.data["e_s"])

        es_update = (
            self.triplet_gate(triplet_message)
            * self.cutoff_fn(edges.data["dist"])[..., None]
        )

        return {"e_s_upd": es_update}

    def _update_edge_v(self, edges: dgl.udf.EdgeBatch):
        vec_channels = self.triplet_vec(edges.data["e_s_upd"])
        triplet_channel, edge_channel, rel_channel = torch.split(
            vec_channels, self.lg_feat_dim, dim=-1
        )

        ev_update = (
            edges.data["e_v"] * triplet_channel
            + edges.src["n_v"] * edge_channel
            + edges.data["vctr_norm"][..., None] * rel_channel
        )
        ev_update *= self.cutoff_fn(edges.data["dist"])[..., None]

        return {"e_v_upd": ev_update}

    def _update_node_v(self, nodes: dgl.udf.NodeBatch):
        n_ev = nodes.data.pop("n_ev")
        n_v_update = self.out_nv(n_ev)
        return {"n_v_upd": n_v_update}

    def _update_node_s(self, nodes: dgl.udf.NodeBatch):
        n_es = nodes.data.pop("n_es")
        n_v_norm = torch.norm(nodes.data["n_v"], dim=1, keepdim=True)
        n_mess = torch.cat([n_es, n_v_norm], dim=-1)
        n_s_update = self.out_ns(n_mess)
        return {"n_s_upd": n_s_update}

    def forward(
        self,
        g: DGLGraph,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ):
        with g.local_scope():
            g.ndata.update({"n_s": self.ns_in(node_s), "n_v": self.nv_in(node_v)})
            g.edata.update({"e_s": edge_s, "e_v": edge_v})

            g.apply_edges(self._update_edge_s)
            g.apply_edges(self._update_edge_v)

            g.update_all(fn.copy_e("e_v_upd", "e_v_upd"), fn.sum("e_v_upd", "n_ev"))
            g.update_all(fn.copy_e("e_s_upd", "e_s_upd"), fn.sum("e_s_upd", "n_es"))

            g.apply_nodes(self._update_node_v)
            g.apply_nodes(self._update_node_s)

            node_s = self.norm_mess(g.ndata.pop("n_s_upd") + node_s)
            node_v = self.norm_nv(g.ndata.pop("n_v_upd") + node_v)
            edge_s = g.edata.pop("e_s_upd") + edge_s
            edge_v = g.edata.pop("e_v_upd") + edge_v

        return node_s, node_v, edge_s, edge_v
