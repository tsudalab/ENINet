import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLGraph

from eninet.model.model import EquiThreeBody


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_random_molecule_graph(num_nodes: int, num_edges: int) -> DGLGraph:
    G = nx.gnm_random_graph(num_nodes, num_edges)
    dgl_graph = dgl.from_networkx(G)

    dgl_graph.ndata["pos"] = torch.randn(num_nodes, 3)
    dgl_graph.ndata["node_type"] = torch.randint(1, 94, (num_nodes,))

    return dgl_graph


def test_equivariance(model: nn.Module, g: DGLGraph, g_feat_dim: int) -> None:
    # Generate random rotation matrix
    rotation_matrix = torch.tensor(np.random.randn(3, 3), dtype=torch.float32)
    rotation_matrix, _ = torch.linalg.qr(
        rotation_matrix
    )  # QR decomposition to get orthogonal matrix

    # Forward pass for original atomic positions
    l_g = g.line_graph(backtracking=False, shared=False)
    output_v = model(g, l_g).ndata["v"]  # [num_nodes, 3, num_channels]

    # Rotate atomic positions and forward pass
    g.ndata["pos"] = torch.matmul(g.ndata["pos"], rotation_matrix)
    l_g = g.line_graph(backtracking=False, shared=False)
    output_v_rot = model(g, l_g).ndata["v"]

    # Rotate output_v
    rotation_matrix_extended = rotation_matrix.unsqueeze(0).expand(g_feat_dim, -1, -1)
    output_v = output_v.permute(0, 2, 1)
    output_v = torch.einsum("ijk,jkl->ijl", output_v, rotation_matrix_extended)
    output_v = output_v.permute(0, 2, 1)

    # assert R @ phi(g) = phi(R @ g)
    assert torch.allclose(output_v_rot, output_v, atol=1e-5)

    print("Equivariance test passed!")


if __name__ == "__main__":

    num_nodes = 10
    num_edges = 20
    g = generate_random_molecule_graph(num_nodes, num_edges)

    g_feat_dim = 5
    model = EquiThreeBody(n_elements=94, g_feat_dim=g_feat_dim)
    test_equivariance(model, g, g_feat_dim)
