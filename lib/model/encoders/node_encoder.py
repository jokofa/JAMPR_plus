#
from warnings import warn
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.nn as gnn

from lib.model.encoders.graph_conv import GraphConvBlock
from lib.model.encoders.eg_graph_conv import EGGConv
from lib.model.encoders.base_encoder import BaseEncoder
from lib.routing import RPObs


class NodeEncoder(BaseEncoder):
    """Graph neural network encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 edge_feature_dim: int = 1,
                 num_layers: int = 3,
                 conv_type: str = "GraphConv",
                 activation: str = "gelu",
                 skip: bool = False,
                 norm_type: Optional[str] = None,
                 add_linear: bool = False,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension of output
            hidden_dim: dimension of hidden layers
            edge_feature_dim: dimension of edge features
            num_layers: number of encoder layers for neighborhood graph
            conv_type: type of graph convolution
            activation: activation function
            skip: flag to use skip (residual) connections
            norm_type: type of norm to use
            add_linear: flag to add linear layer after conv
        """
        super(NodeEncoder, self).__init__(input_dim, output_dim, hidden_dim)

        self.num_layers = num_layers
        if edge_feature_dim is not None and edge_feature_dim != 1 and conv_type.upper() != "EGGCONV":
            raise ValueError("encoders currently only work for edge_feature_dim=1")
        self.edge_feature_dim = edge_feature_dim

        self.conv_type = conv_type
        self.activation = activation
        self.skip = skip
        self.norm_type = norm_type
        self.add_linear = add_linear
        self.eggc = False

        self.input_proj = None
        self.input_proj_e = None
        self.output_proj = None
        self.layers = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self._reset_module_list(self.layers)
        if self.input_proj_e is not None:
            self.input_proj_e.reset_parameters()

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input projection layer
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        if self.conv_type.upper() == "EGGCONV":
            # special setup for EGGConv which propagates node AND edge embeddings
            self.eggc = True
            if self.activation.lower() != 'relu':
                warn(f"EGGConv normally uses RELU but got {self.activation.upper()}")
            if self.norm_type is None:
                self.norm_type = "bn"
            elif self.norm_type.lower() not in ['bn', 'batch_norm']:
                warn(f"EGGConv normally uses BN but got {self.norm_type.upper()}")
            self.input_proj_e = nn.Linear(self.edge_feature_dim, self.hidden_dim)

            def GNN():
                return EGGConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    activation=self.activation,
                    norm_type=self.norm_type,
                )
        else:
            conv = getattr(gnn, self.conv_type)

            def GNN():
                # creates a GNN module with specified parameters
                # all modules are initialized globally with the call to
                # reset_parameters()
                return GraphConvBlock(
                        conv,
                        self.hidden_dim,
                        self.hidden_dim,
                        activation=self.activation,
                        skip=self.skip,
                        norm_type=self.norm_type,
                        add_linear=self.add_linear,
                        **kwargs
                )

        # nbh based node embedding layers
        if self.num_layers > 0:
            self.layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.layers.append(GNN())

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, obs: RPObs, **kwargs) -> Tensor:

        bs = obs.batch_size
        x = obs.node_features
        nbh_e = obs.node_nbh_edges
        nbh_w = obs.node_nbh_weights

        x = x.view(-1, x.size(-1))
        # input layer
        x = self.input_proj(x)

        # encode nbh node embeddings
        assert nbh_e is not None and nbh_w is not None
        if self.eggc:
            nbh_w = self.input_proj_e(nbh_w[:, None])
        for layer in self.layers:
            x, nbh_w = layer(x, nbh_e, nbh_w)

        # output layer
        x = self.output_proj(x)

        # check for NANs
        if (x != x).any():
            raise RuntimeError(f"Output includes NANs! (e.g. GCNConv can produce NANs when <normalize=True>!)")

        # reshape to (bs, n, d) - this simplifies downstream processing
        # but will not work for batches with differently sized graphs (!)
        return x.view(bs, -1, self.output_dim)


# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 5,
        n: int = 10,
        cuda=False,
        seed=1
):
    import sys
    from lib.utils.graph_utils import GraphNeighborhoodSampler
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # testing args
    num_layers = [1, 2, 3]
    conv_types = ["EGGConv", "GCNConv", "GraphConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "ClusterGCNConv"]
    norm_types = [None, "ln", "bn"]
    skips = [True, False]
    add_lins = [True, False]

    # create data
    I = 4
    O = 32
    x = torch.randn(bs, n, I).to(device)

    # sample edges and weights
    sampler = GraphNeighborhoodSampler(graph_size=n, k_frac=0.5)
    coords = x[:, :, -2:]
    edge_idx, edge_weights = [], []
    for c in coords:
        ei = sampler(c)
        edge_idx.append(ei)
        idx_coords = c[ei]
        edge_weights.append(
            torch.norm(idx_coords[0]-idx_coords[1], p=2, dim=-1)
        )
    reps = edge_idx[0].size(-1)
    edge_idx = torch.stack(edge_idx, dim=0).permute(1, 0, 2).reshape(2, -1)
    # transform to running idx
    idx_inc = (torch.cumsum(torch.tensor([n]*bs), dim=0) - n).repeat_interleave(reps)
    edge_idx += idx_inc
    edge_weights = torch.stack(edge_weights).view(-1)

    x = RPObs(
        batch_size=bs,
        node_features=x.view(-1, I),
        node_nbh_edges=edge_idx,
        node_nbh_weights=edge_weights,
        tour_plan=None,
        tour_features=None,
        tour_edges=None,
        tour_weights=None,
        nbh=None,
        nbh_mask=None,
    )

    for l in num_layers:
        for c_type in conv_types:
            for norm in norm_types:
                for skip in skips:
                    for add_lin in add_lins:
                        try:
                            e = NodeEncoder(
                                I, O,
                                num_layers=l,
                                conv_type=c_type,
                                norm_type=norm,
                                skip=skip,
                                add_linear=add_lin
                            ).to(device)
                            out = e(x)
                            assert out.size() == torch.empty((bs, n, O)).size()
                        except Exception as e:
                            raise type(e)(
                                str(e) + f" - ("
                                         f"num_layers: {l}, "
                                         f"conv_type: {c_type}, "
                                         f"norm: {norm}, "
                                         f"skip: {skip}, "
                                         f"add_lin: {add_lin}, "
                                         f")\n"
                            ).with_traceback(sys.exc_info()[2])

