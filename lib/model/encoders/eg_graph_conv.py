#
from typing import Tuple
from torch_geometric.typing import Adj

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_sum
from torch_geometric.nn.conv import MessagePassing

from lib.utils import get_activation_fn, get_norm


#
class EGGConv(MessagePassing):
    """Gated graph convolution using node and edge information
    ('edge gated graph convolution' - EGGC).

     torch geometric implementation based on original formulation in
        - Bresson and Laurent 2018, Residual Gated Graph ConvNets
        - Joshi et al. 2019, An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = "relu",
                 norm_type: str = "bn",
                 bias: bool = False,
                 aggr: str = "mean",
                 **kwargs):
        super(EGGConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_type = norm_type
        self.act = get_activation_fn(activation, module=True, inplace=False)
        assert in_channels == out_channels, f"currently only works for 'in_channels' == 'out_channels'"

        self.w1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.w2 = nn.Linear(in_channels, out_channels, bias=bias)
        self.w3 = nn.Linear(in_channels, out_channels, bias=bias)
        self.w4 = nn.Linear(in_channels, out_channels, bias=bias)
        self.w5 = nn.Linear(in_channels, out_channels, bias=bias)

        self.edge_norm = get_norm(norm_type, hdim=out_channels)
        self.node_norm = get_norm(norm_type, hdim=out_channels)

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()
        self.w4.reset_parameters()
        self.w5.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                e: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        # message passing
        new_x, new_e = self.propagate(edge_index, x=x, e_ij=e)
        # apply BN and activation
        x = x + self.act(self.node_norm(self.w1(x) + new_x))
        e = e + self.act(self.edge_norm(new_e))
        return x, e

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                e_ij: Tensor,
                index: Tensor
                ) -> Tuple[Tensor, Tensor]:

        # calculate node proj
        w2x_j = self.w2(x_j)
        # calculate gates
        eta_ij = torch.sigmoid(e_ij)
        gated_x_j = eta_ij * w2x_j
        # aggregate
        if self.aggr == 'mean':
            # rather than a mean this is normalizing the gates!
            aggr_x = scatter_sum(gated_x_j, index=index, dim=0) / (1e-20 + scatter_sum(eta_ij, index=index, dim=0))
        elif self.aggr == 'sum':
            aggr_x = scatter_sum(gated_x_j, index=index, dim=0)
        else:
            raise RuntimeError(f"aggregation {self.aggr} not supported.")

        # calculate edge proj
        w3e_ij = self.w3(e_ij)
        w4x_i = self.w4(x_i)
        w5x_j = self.w5(x_j)
        # new edge emb
        e_ij = w3e_ij + w4x_i + w5x_j

        return aggr_x, e_ij

    def aggregate(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # overwrite with pass through identity,
        # since aggregation is already done in message()
        return inputs

    def __repr__(self):
        return '{}(in: {}, out: {}, act_fn: {}, norm: {})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.activation,
            self.norm_type,
        )


#
# ============= #
# ### TEST #### #
# ============= #
def _test(
        cuda=False,
        seed=1,
):
    import sys
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    N = 4
    D = 16
    x = torch.randn(N, D).to(device)
    edge_index = torch.tensor([[0, 1, 2, 2, 3, 3], [0, 0, 1, 1, 3, 2]]).to(device)
    edge_weight = torch.randn(edge_index.size(-1), D).to(device)
    conv = EGGConv(D, D)

    try:
        x, e = conv(x, edge_index, edge_weight)
        assert x.size() == (N, D)
        assert e.size() == (edge_index.size(-1), D)
    except Exception as e:
        raise type(e)(str(e)).with_traceback(sys.exc_info()[2])

