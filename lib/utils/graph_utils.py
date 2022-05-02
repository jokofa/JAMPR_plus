#
from __future__ import annotations
from typing import Tuple
from torch_geometric.typing import Adj

import math
import numpy as np
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_cluster import knn_graph


def flip_lr(x: LongTensor):
    """
    Flip the first dimension in left-right direction.
    This is used to reverse tours by swapping
    (from, to) edges to (to, from) format.
    """
    return torch.fliplr(x.unsqueeze(0)).squeeze(0)


def to_sparse(adj: np.ndarray) -> np.ndarray:
    """Takes a numpy array adjacency matrix and converts it to
    sparse format as array of edge indices."""
    adj = torch.from_numpy(adj)
    adj, _ = dense_to_sparse(adj)
    return adj.numpy()


def slice_tensor(x: Tensor, slice_range: Tuple[int, int], dim: int = -1):
    """Slice the given tensor over provided index slice_range in specified dim."""
    s, e = slice_range
    idx = torch.arange(x.size(dim), device=x.device)
    idx = idx[(s <= idx) & (idx < e)]
    return torch.index_select(x, dim, idx)


def negative_nbh_sampling(edge_index: Adj,
                          max_k: int,
                          num_neg_samples: int,
                          loop: bool = False) -> Adj:
    """
    Takes a sparse neighborhood adjacency matrix and
    adds <num_neg_samples> random edges for each node.
    """
    _, n, k = edge_index.size()
    # possible range of indices
    idx_range = torch.arange(max_k, device=edge_index.device)
    # get indices not yet in edge_index
    mask = ~(
        edge_index[0][:, :, None].expand(n, k, max_k)
        ==
        idx_range[None, None, :].expand(n, k, max_k)
    ).any(dim=1)
    # mask same node indices (self loops)
    if not loop:
        mask &= (edge_index[1, :, 0][:, None].expand(-1, max_k) != idx_range[None, :].expand(n, max_k))
    # get candidate indices
    candidates = idx_range[None, :].expand(n, -1)[mask].view(n, -1)
    # sample idx and create edge
    i = int(not loop)  # i = 1 when not considering self loops!
    return torch.cat(
        (candidates[:, torch.randperm(max_k-k-i)[:num_neg_samples]].unsqueeze(0),
         edge_index[1, :, 0][:, None].expand(-1, num_neg_samples).unsqueeze(0)),
        dim=0
    )


class GraphNeighborhoodSampler(nn.Module):
    def __init__(self,
                 graph_size: int,
                 k_frac: float = 0.3,
                 rnd_edge_ratio: float = 0.0,
                 num_workers: int = 4,
                 **kwargs):
        """Samples <k_frac> nearest neighbors +
        <rnd_edge_ratio> random nodes as initial graph.

        Args:
            graph_size: size of considered graph
            k_frac: number of neighbors considered
            rnd_edge_ratio: ratio of random edges among neighbors
                            to have connections beyond local neighborhood
            num_workers: number of workers
            **kwargs:
        """
        super(GraphNeighborhoodSampler, self).__init__()
        self.graph_size = graph_size
        self.k_frac = k_frac
        self.rnd_edge_ratio = rnd_edge_ratio
        self.num_workers = num_workers if torch.cuda.is_available() else 1
        self.k, self.max_k, self.k_nn, self.num_rnd = None, None, None, None
        self._infer_k(graph_size)

    def _infer_k(self, n: int):
        self.max_k = n
        self.k = math.ceil(self.k_frac * self.max_k)
        # infer how many neighbors are nodes sampled randomly from graph
        self.num_rnd = math.floor(self.k * self.rnd_edge_ratio)
        self.k_nn = self.k - self.num_rnd

    @torch.no_grad()
    def forward(self, coords: Tensor, loop: bool = True):
        n, d = coords.size()
        if n != self.graph_size:
            self._infer_k(n)

        # remove depot coords
        coords = coords[1:, :].view(-1, d)
        # get k nearest neighbors
        edge_idx = knn_graph(coords,
                             k=self.k_nn-1,     # since we add depot to each node nbh
                             loop=loop,     # include self-loops flag
                             num_workers=self.num_workers)
        # sample additional edges to random nodes if specified
        if self.num_rnd > 0:
            edge_idx = edge_idx.view(2, -1, self.k_nn-1)
            rnd_edge_idx = negative_nbh_sampling(edge_index=edge_idx,
                                                 max_k=self.max_k,
                                                 num_neg_samples=self.num_rnd,
                                                 loop=False)
            edge_idx = torch.cat((edge_idx, rnd_edge_idx), dim=-1).view(2, -1)

        # add depot node into nbh of each node and vice versa
        to_depot_edges = torch.cat((
            torch.arange(n, device=coords.device)[None, :],
            torch.zeros(n, dtype=torch.long, device=coords.device)[None, :]
        ), axis=0)
        from_depot_edges = flip_lr(to_depot_edges[:, 1:])

        edge_idx = torch.cat((
            edge_idx.view(2, n - 1, -1) + 1,
            from_depot_edges[:, :, None]
        ), dim=-1)

        return torch.cat((
            to_depot_edges,
            edge_idx.view(2, -1)
        ), dim=-1)


#
# ============= #
# ### TEST #### #
# ============= #
def _test_slice(cuda=False, seed=1):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    t = torch.randn((4, 10, 4)).to(device)
    idx_tuple = (1, 3)

    s, e = idx_tuple
    idx = torch.arange(t.size(-1)).to(device)
    idx = idx[(s <= idx) & (idx < e)]
    t2 = t[:, :, idx]
    t3 = slice_tensor(t, idx_tuple)
    assert (t2 == t3).all()
