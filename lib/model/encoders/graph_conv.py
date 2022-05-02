#
from typing import Optional, Tuple
from torch_geometric.typing import Adj

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

from lib.utils import get_activation_fn, get_norm


EDGECONVS = [
    # edge_weight:
    "GCNConv", "GCN2Conv", "GraphConv", "GatedGraphConv",
    "TAGConv", "SGConv", "FAConv", "APPNP", "ARMAConv",
    # edge_attr:
    "TransformerConv", "GINEConv", "GMMConv", "GENConv", "GATv2Conv",
]


#
# NOTE: just for the record - we use the following ordering on layers: (Conv/Lin -> Act -> Norm)
#
class GraphConvBlock(nn.Module):
    """
    Full graph convolutional block including
    convolution and activation as well as optional
    norm, skip connection and added linear layer.
    """
    def __init__(self,
                 conv: MessagePassing,
                 in_channels: int,
                 out_channels: int,
                 activation: str = "gelu",
                 skip: bool = True,
                 norm_type: Optional[str] = "ln",
                 add_linear: bool = False,
                 aggr: str = "max",
                 **kwargs
                 ):
        super(GraphConvBlock, self).__init__()
        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            **kwargs
        )
        self.activation = activation
        self.act = get_activation_fn(activation, module=True, **kwargs)
        self.norm_type = norm_type
        self.norm = get_norm(norm_type, hdim=out_channels, **kwargs)
        self.skip = skip
        if self.skip and in_channels != out_channels:
            raise RuntimeError(f"To apply skip connection, in_channels and out_channels must be the same!")
        self.add_linear = add_linear
        self.lin = nn.Linear(out_channels, out_channels) if add_linear else None
        self.lin_norm = get_norm(norm_type, hdim=out_channels, **kwargs) if add_linear else None
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.add_linear:
            self.lin.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: Optional[Tensor] = None,
                **kwargs) -> Tuple[Tensor, Tensor]:
        if self.skip:
            x_ = x
        if self.conv.__class__.__name__ in EDGECONVS:
            # provide additional edge weights / attributes
            x = self.act(self.conv(x, edge_index, edge_weight, **kwargs))
        else:
            x = self.act(self.conv(x, edge_index, **kwargs))
        if self.skip:
            x += x_
        if self.norm is not None:
            x = self.norm(x)
        if self.add_linear:
            if self.skip:
                x_ = x
            x = self.act(self.lin(x))
            if self.skip:
                x += x_
            if self.lin_norm is not None:
                x = self.norm(x)
        return x, edge_weight

    def __repr__(self):
        return '{}(conv={}, in={}, out={}, act_fn={}, norm={}, skip={}, add_linear={})'.format(
            self.__class__.__name__,
            self.conv.__class__.__name__,
            self.conv.in_channels,
            self.conv.out_channels,
            self.act.__class__.__name__,
            self.norm_type,
            self.skip,
            self.add_linear,
        )


#
# ============= #
# ### TEST #### #
# ============= #
def _test(
        cuda=False,
        seed=1,
        **kwargs
):
    import sys
    import torch_geometric.nn as gnn
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    conv_types_with_ew = ["GCNConv", "GraphConv"]
    conv_types_without_ew = ["ResGatedGraphConv", "GATConv", "GATv2Conv", "ClusterGCNConv"]
    conv_types = conv_types_with_ew + conv_types_without_ew
    norm_types = [None, "ln", "bn"]

    D = 16
    x = torch.randn(4, D).to(device)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]).to(device)
    edge_weight = torch.randn(edge_index.size(-1)).to(device)

    for c_type in conv_types:
        for norm in norm_types:
            try:
                c_ = getattr(gnn, c_type)
                conv = GraphConvBlock(c_, D, D, norm_type=norm).to(device)
                if c_type in conv_types_with_ew:
                    out, _ = conv(x, edge_index, edge_weight)
                else:
                    out, _ = conv(x, edge_index)
                assert out.size() == (4, D)
            except Exception as e:
                raise type(e)(str(e)+f" - (conv: {c_type}, norm: {norm})\n").with_traceback(sys.exc_info()[2])


