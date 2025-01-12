from typing import Dict, List, Optional, Tuple

import dgl
import torch
from dgl import DGLGraph
from torch import Tensor


def collate_fn_lg(
    batch: List[Tuple[DGLGraph, Optional[DGLGraph], Tensor]]
) -> Tuple[DGLGraph, DGLGraph, Tensor]:
    """Merge a list of dgl graphs to form a batch."""
    graphs, linegraphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)

    # if linegraph is None, create linegraph
    if linegraphs[0] is None:
        l_g = dgl.batch(
            [graph.line_graph(backtracking=False, shared=False) for graph in graphs]
        )
    else:
        l_g = dgl.batch(linegraphs)
    labels = torch.stack(labels, dim=0)
    return g, l_g, labels


def collate_fn_g(
    batch: List[Tuple[DGLGraph, Optional[DGLGraph], Tensor]]
) -> Tuple[DGLGraph, None, Tensor]:
    """Merge a list of dgl graphs to form a batch."""
    graphs, _, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)
    return g, None, labels


def collate_fn_lg_pes(
    batch: List[Tuple[DGLGraph, Optional[DGLGraph], Tensor]]
) -> Tuple[DGLGraph, DGLGraph, Dict]:
    """Merge a list of dgl graphs to form a batch."""
    graphs, linegraphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    if linegraphs[0] is None:
        l_g = dgl.batch(
            [graph.line_graph(backtracking=False, shared=False) for graph in graphs]
        )
    else:
        l_g = dgl.batch(linegraphs)
    labels = {
        "E": torch.stack([l["E"] for l in labels], dim=0),
        "F": torch.stack([l["F"] for l in labels], dim=0),
    }
    return g, l_g, labels


def collate_fn_g_pes(
    batch: List[Tuple[DGLGraph, Optional[DGLGraph], Tensor]]
) -> Tuple[DGLGraph, None, Dict]:
    """Merge a list of dgl graphs to form a batch."""
    graphs, _, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = {
        "E": torch.stack([l["E"] for l in labels], dim=0),
        "F": torch.stack([l["F"] for l in labels], dim=0),
    }
    return g, None, labels
