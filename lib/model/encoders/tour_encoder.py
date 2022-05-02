#
from warnings import warn
from typing import Optional
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
import torch_geometric.nn as gnn

from lib.model.encoders.graph_conv import GraphConvBlock
from lib.model.encoders.eg_graph_conv import EGGConv
from lib.model.encoders.base_encoder import BaseEncoder
from lib.routing import RPObs
from lib.utils.graph_utils import flip_lr


class TourEncoder(BaseEncoder):
    """Graph neural network encoder model for node embeddings."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 node_emb_dim: int,
                 hidden_dim: int = 128,
                 edge_feature_dim: int = 1,
                 num_layers: int = 2,
                 propagate_reverse: bool = False,
                 consolidate_nbh: bool = True,
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
            num_layers: number of encoder layers for tour graphs
            propagate_reverse: flag to also propagate in reverse tour direction
            consolidate_nbh: flag to re-propagate over nbh graph after each tour propagation
                             (adds a new GNN layer for each re-propagation)
            conv_type: type of graph convolution
            activation: activation function
            skip: flag to use skip (residual) connections
            norm_type: type of norm to use
            add_linear: flag to add linear layer after conv
        """
        super(TourEncoder, self).__init__(input_dim, output_dim, hidden_dim)

        self.node_emb_dim = node_emb_dim
        self.num_layers = num_layers
        self.propagate_reverse = propagate_reverse
        self.consolidate_nbh = consolidate_nbh
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
        self.node_emb_input_proj = None
        self.input_proj_e = None
        self.output_proj = None
        self.node_emb_output_proj = None
        self.nbh_layers = None
        self.tour_layers = None
        self.rev_tour_layers = None
        self.best_tour_layers = None
        self.best_tour_rev_layers = None

        self.static_tour_node_emb = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.output_proj.reset_parameters()
        self.node_emb_output_proj.reset_parameters()
        self._reset_module_list(self.tour_layers)
        self._reset_module_list(self.rev_tour_layers)
        if self.input_proj_e is not None:
            self.input_proj_e.reset_parameters()
        if self.node_emb_input_proj is not None:
            self.node_emb_input_proj.reset_parameters()

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # input layers
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        if self.node_emb_dim != self.hidden_dim:
            self.node_emb_input_proj = nn.Linear(self.node_emb_dim, self.hidden_dim)

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

        inc = int(self.consolidate_nbh)
        # tour embedding layers
        self.tour_layers = nn.ModuleList()
        for _ in range(self.num_layers + inc):
            self.tour_layers.append(GNN())

        if self.propagate_reverse:
            self.rev_tour_layers = nn.ModuleList()
            for _ in range(self.num_layers + inc):
                self.rev_tour_layers.append(GNN())

        # output layers
        self.output_proj = nn.Linear(2*self.hidden_dim, self.output_dim)
        self.node_emb_output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, obs: RPObs, static_node_emb: Tensor = None, **kwargs):
        bs, nf, nbh_e, nbh_w, active_plan, tf, tour_e, tour_w, _, _ = obs

        # encode node embeddings over tours
        if len(tour_e) == 0 and len(tour_w) == 0:
            # use static emb, no new information added via tour graph
            assert self.static_tour_node_emb is not None
            node_emb = self.static_tour_node_emb
        else:
            assert tour_e is not None and tour_w is not None
            node_emb = static_node_emb
            if node_emb.size(-1) != self.hidden_dim:
                node_emb = self.node_emb_input_proj(node_emb)
            node_emb = node_emb.view(-1, node_emb.size(-1))

            if self.eggc:
                if len(nbh_w.shape) == 1:
                    nbh_w = self.input_proj_e(nbh_w[:, None])
                tour_w = self.input_proj_e(tour_w[:, None])
            for i, layer in enumerate(self.tour_layers):
                if self.consolidate_nbh and i == len(self.tour_layers) - 1:
                    # last layer aggregates again over nbh
                    node_emb, nbh_w = layer(node_emb, nbh_e, nbh_w)
                else:
                    node_emb, tour_w = layer(node_emb, tour_e, tour_w)
            if self.propagate_reverse:
                # reverse tour indices
                rev_tour_e = flip_lr(tour_e)
                for i, layer in enumerate(self.rev_tour_layers):
                    if self.consolidate_nbh and i == len(self.rev_tour_layers) - 1:
                        # last layer aggregates again over nbh
                        node_emb, nbh_w = layer(node_emb, nbh_e, nbh_w)
                    else:
                        node_emb, cur_tour_w = layer(node_emb, rev_tour_e, tour_w)

            # check for NANs
            if (node_emb != node_emb).any():
                raise RuntimeError(f"Output includes NANs! (e.g. GCNConv can produce NANs when <normalize=True>!)")
            # save static
            self.static_tour_node_emb = node_emb.clone()

        _, k, _ = tf.size()
        # project tour features
        tour_emb = self.input_proj(tf)

        # cat tour emb components and feed through output layer
        tour_emb = self.output_proj(torch.cat((
            tour_emb,
            # per_tour_emb
            self._compute_per_tour_emb(
                node_emb=node_emb,
                active_plan=active_plan,
                bs=bs, k=k, hidden_dim=self.hidden_dim
            )
        ), dim=-1))

        # reshape to (bs, n, d) - this simplifies downstream processing
        # but will not work for batches with differently sized graphs (!)
        node_emb = self.node_emb_output_proj(node_emb.view(bs, -1, self.hidden_dim))
        node_emb = node_emb + static_node_emb   # skip connection

        return tour_emb, node_emb

    def reset_static(self):
        """Reset buffers of static components."""
        self.static_tour_node_emb = None

    @staticmethod
    @torch.jit.script
    def _compute_per_tour_emb(node_emb: Tensor, active_plan: Tensor, bs: int, k:int, hidden_dim: int):
        """Compute the emb on a per tour basis for all currently active tours."""
        node_emb = node_emb.view(bs, -1, hidden_dim)
        # select node_emb according to tours
        per_tour_emb = node_emb[:, None, :, :].expand(bs, k, -1, hidden_dim).gather(
            dim=2, index=active_plan[:, :, :, None].expand(
                active_plan.size(0), active_plan.size(1), active_plan.size(2), hidden_dim)
        )
        # get idx of return to depot (end of tour)
        # get mean of per_tour_emb by selecting cumsum of emb at idx and dividing by len (idx+1=len)
        idx = (active_plan > 0).to(torch.int).argmin(dim=-1)
        # per_tour_emb_cmsm = torch.cumsum(per_tour_emb, dim=2)
        return torch.cumsum(per_tour_emb, dim=2).gather(
            dim=2,
            index=idx[:, :, None, None].expand(bs, k, 1, hidden_dim)
        ).view(bs, k, hidden_dim) / (idx + 1)[:, :, None]



# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 3,
        n: int = 20,
        cuda=False,
        seed=1
):
    import sys
    import math
    import numpy as np
    from lib.utils.graph_utils import GraphNeighborhoodSampler
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rnd = np.random.default_rng(seed)

    # testing args
    num_layers = [1, 2, 3]
    conv_types = ["EGGConv", "GCNConv", "GraphConv", "ResGatedGraphConv", "GATConv", "GATv2Conv", "ClusterGCNConv"]
    norm_types = [None, "ln", "bn"]
    skips = [True, False]
    add_lins = [True, False]
    prop_rev = [True, False]
    consolidate = [True, False]

    # create data
    I = 4
    O = 32
    x = torch.randn(bs, n, O).to(device)
    n_tours = math.floor(math.sqrt(n))

    # sample edges and weights of NBH
    sampler = GraphNeighborhoodSampler(graph_size=n, k_frac=0.5)
    coords = x[:, :, -2:]
    edge_idx, edge_weights = [], []
    for c in coords:
        ei = sampler(c)
        edge_idx.append(ei)
        idx_coords = c[ei]
        edge_weights.append(
            torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)
        )
    reps = edge_idx[0].size(-1)
    edge_idx = torch.stack(edge_idx, dim=0).permute(1, 0, 2).reshape(2, -1)
    # transform to running idx
    idx_inc = (torch.cumsum(torch.tensor([n] * bs), dim=0) - n).repeat_interleave(reps)
    edge_idx += idx_inc
    edge_weights = torch.stack(edge_weights).view(-1)

    # sample tours
    tour_edge_idx, tour_edge_weights, tour_plans = [], [], []
    for c in coords:
        plan = torch.zeros((n_tours, n//2), dtype=torch.long)
        indices = np.arange(1, n)
        rnd.shuffle(indices)
        per_tour = math.floor((n-1)/n_tours)
        last_tour = per_tour + ((n-1)-n_tours*per_tour)
        indices = torch.from_numpy(indices)
        idx = torch.cumsum(torch.tensor([0] + [per_tour]*(n_tours-1) + [last_tour], dtype=torch.long), dim=-1)
        for i, (s, e) in enumerate(zip(idx[:-1], idx[1:])):
            slice = indices[s:e]
            plan[i, :len(slice)] = slice
        plan = plan.to(device)
        tour_plans.append(plan)
        ei = torch.cat((
            torch.roll(plan, shifts=1, dims=-1)[:, None, :],   # cyclic shift by 1
            plan[:, None, :]
        ), axis=1).permute(1, 0, 2).reshape(2, -1)
        selection_mask = (ei[0, :] != ei[1, :])
        ei = ei[:, selection_mask]
        tour_edge_idx.append(ei)
        idx_coords = c[ei]
        tour_edge_weights.append(
            torch.norm(idx_coords[0] - idx_coords[1], p=2, dim=-1)
        )
    reps = tour_edge_idx[0].size(-1)
    tour_edge_idx = torch.stack(tour_edge_idx, dim=0).permute(1, 0, 2).reshape(2, -1)
    # transform to running idx
    idx_inc = (torch.cumsum(torch.tensor([n] * bs), dim=0) - n).repeat_interleave(reps)
    tour_edge_idx += idx_inc
    tour_edge_weights = torch.stack(tour_edge_weights).view(-1)
    tour_plans = torch.stack(tour_plans)
    n_emb = x
    tf = torch.randn(bs, n_tours, I).to(device)

    x = RPObs(
        batch_size=bs,
        node_features=x.view(-1, I),
        node_nbh_edges=edge_idx,
        node_nbh_weights=edge_weights,
        tour_plan=tour_plans,
        tour_features=tf,
        tour_edges=tour_edge_idx,
        tour_weights=tour_edge_weights,
        nbh=None,
        nbh_mask=None,
    )

    for l in num_layers:
        for c_type in conv_types:
            for norm in norm_types:
                for skip in skips:
                    for add_lin in add_lins:
                        for pr in prop_rev:
                            for con in consolidate:
                                try:
                                    e = TourEncoder(
                                        I, O,
                                        node_emb_dim=O,
                                        num_layers=l,
                                        propagate_reverse=pr,
                                        consolidate_nbh=con,
                                        conv_type=c_type,
                                        norm_type=norm,
                                        skip=skip,
                                        add_linear=add_lin
                                    ).to(device)
                                    t_emb, new_n_emb = e(x, n_emb)
                                    assert t_emb.size() == torch.empty((bs, n_tours, O)).size()
                                    assert new_n_emb.size() == n_emb.size()
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

