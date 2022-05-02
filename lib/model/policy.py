#
import logging
from typing import Optional, Dict, Tuple, Any, NamedTuple, Union
from torch import Tensor, BoolTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from lib.routing import RPObs
from lib.utils import count_parameters
import lib.model.encoders as _encoders
import lib.model.decoders as _decoders

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    """
    Model wrapping encoder and decoder models.

    Args:
        observation_space: env observation space
        node_encoder_type: type of node encoder
        node_encoder_args: additional arguments for encoder creation
        tour_encoder_type: type of tour encoder
        tour_encoder_args: additional arguments for encoder creation
        decoder_type: type of decoder
        decoder_args: additional arguments for decoder creation
        embedding_dim: general embedding dimension of model
        device: device of model (CPU or GPU)

    """

    def __init__(self,
                 observation_space: Dict,
                 node_encoder_type: Union[str, nn.Module] = "NodeEncoder",
                 node_encoder_args: Optional[Dict] = None,
                 tour_encoder_type: Union[str, nn.Module] = "TourEncoder",
                 tour_encoder_args: Optional[Dict] = None,
                 decoder_type: Union[str, nn.Module] = "AttnDecoder",
                 decoder_args: Optional[Dict] = None,
                 embedding_dim: int = 256,
                 device: Union[str, int, torch.device] = "cpu",
                 **kwargs):
        super(Policy, self).__init__()

        self.observation_space = observation_space
        self.node_encoder_type = node_encoder_type
        self.node_encoder_args = node_encoder_args if node_encoder_args is not None else {}
        self.tour_encoder_type = tour_encoder_type
        self.tour_encoder_args = tour_encoder_args if tour_encoder_args is not None else {}
        self.decoder_type = decoder_type
        self.decoder_args = decoder_args if decoder_args is not None else {}
        self.embedding_dim = embedding_dim
        self._device = torch.device(device)

        # get dims from obs and act space
        self.node_feature_dim = self.observation_space['node_features']
        self.tour_feature_dim = self.observation_space['tour_features']

        # initialize encoder models
        n_enc_cl = getattr(_encoders, node_encoder_type) if isinstance(node_encoder_type, str) else node_encoder_type
        self.node_encoder = n_enc_cl(
            input_dim=self.node_feature_dim,
            output_dim=embedding_dim,
            edge_feature_dim=1,
            **self.node_encoder_args, **kwargs)

        t_enc_cl = getattr(_encoders, tour_encoder_type) if isinstance(tour_encoder_type, str) else tour_encoder_type
        self.tour_encoder = t_enc_cl(
            input_dim=self.tour_feature_dim,
            output_dim=embedding_dim,
            node_emb_dim=embedding_dim,
            edge_feature_dim=1,
            **self.tour_encoder_args, **kwargs)

        query_dim = 2 * embedding_dim
        action_dim = embedding_dim

        # initialize decoder model
        dec_cl = getattr(_decoders, decoder_type) if isinstance(decoder_type, str) else decoder_type
        self.decoder = dec_cl(
            query_emb_dim=query_dim,
            action_emb_dim=action_dim,
            **self.decoder_args, **kwargs
        )

        self.static_node_emb = None

        self.reset_parameters()
        self.to(device=self._device)

        self.decode_type = "greedy"

    def __repr__(self):
        super_repr = super().__repr__()  # get torch module str repr
        n_enc_p = count_parameters(self.node_encoder)
        t_enc_p = count_parameters(self.tour_encoder)
        dec_p = count_parameters(self.decoder)
        add_repr = f"\n-----------------------------------" \
                   f"\nNum Parameters: " \
                   f"\n  (node_encoder): {n_enc_p} " \
                   f"\n  (tour_encoder): {t_enc_p} " \
                   f"\n  (decoder): {dec_p} " \
                   f"\n  total: {n_enc_p + t_enc_p + dec_p}"
        return super_repr + add_repr

    @property
    def device(self):
        return self._device

    def reset_parameters(self):
        """Reset model parameters."""
        self.node_encoder.reset_parameters()
        self.tour_encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self,
                obs: RPObs,
                recompute_static: bool = False,
                ) -> Tuple[Tensor, Tensor, Tensor]:

        """Default forward pass

        Args:
            obs: observation batch
            recompute_static: flag to recompute static embedding components

        Returns:
            action, log_likelihood, entropy

        """
        if recompute_static or self.static_node_emb is None:
            self.static_node_emb = self.node_encoder(obs)
        tour_emb, node_emb = self.tour_encoder(obs, self.static_node_emb)
        return self._decoder_forward(obs, node_emb, tour_emb)

    def _decoder_forward(self,
                         obs: RPObs,
                         node_emb: Tensor,
                         tour_emb: Tensor,
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """One step of decoding procedure
        
        Args:
            obs: observation batch
            node_emb: static node embeddings produced by node encoder
            tour_emb: tour embedding produced by tour encoder

        Returns:
            actions, logits, action_idx, entropy
        """

        query_emb = self._create_query_emb(node_emb, tour_emb)
        action_emb, mask = self._create_action_emb(obs, node_emb, tour_emb)

        logits = self.decoder(query_emb, action_emb, mask)
        return self._select_a(obs, logits)

    def reset_static(self):
        """Reset buffers of static components."""
        self.static_node_emb = None
        self.tour_encoder.reset_static()

    ###
    # TODO: cat or sum embeddings???
    ###
    @staticmethod
    @torch.jit.script
    def _create_query_emb(node_emb: Tensor,
                          tour_emb: Tensor,
                          ) -> Tensor:
        """Create the embedding of the query (context)."""
        # graph emb -> aggregate over node dim
        # plan emb -> aggregate over tour dim
        # sum graph and plan embeddings
        return torch.cat((
            node_emb.mean(dim=1) + tour_emb.mean(dim=1),
            node_emb.max(dim=1).values + tour_emb.max(dim=1).values,
        ), dim=-1)

    @staticmethod
    @torch.jit.script
    def _create_action_emb(obs: RPObs,
                           node_emb: Tensor,
                           tour_emb: Tensor,
                           ) -> Tuple[Tensor, BoolTensor]:
        """Create the embeddings of the combinatorial actions and the corresponding mask."""
        bs, k, d = tour_emb.size()
        n = node_emb.size(1)
        nbh_size = obs.nbh.size(-1)
        # select embeddings according to specified nbh
        nbh_emb = node_emb[:, None, :, :].expand(bs, k, n, d).gather(
            dim=2, index=obs.nbh[:, :, :, None].expand(bs, k, nbh_size, d)
        )
        return (    # sum tour_emb and nbh_emb to action embedding
            (tour_emb[:, :, None, :].expand(bs, k, nbh_size, d) + nbh_emb).view(bs, -1, d),
            obs.nbh_mask.view(bs, -1)   # action mask from env
        )

    def _select_a(self,
                  obs: RPObs,
                  logits: Tensor
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select an action given the logits from the decoder."""
        assert (logits == logits).all(), "Logits contain NANs!"
        ent = None
        if self.decode_type == "greedy":
            a_idx = logits.argmax(dim=-1)
        elif self.decode_type == "sampling":
            dist = ActionDist(logits=logits)
            a_idx = dist.sample()
            ent = dist.entropy()
        else:
            raise RuntimeError(f"Unknown decoding type <{self.decode_type}> (Forgot to 'set_decode_type()' ?)")

        bs, k, nbh_size = obs.nbh.size()
        # infer action
        tr = torch.div(a_idx, nbh_size, rounding_mode='floor')  # a_idx//nbh_size
        nd = obs.nbh.view(bs, -1)[torch.arange(bs, device=logits.device), a_idx]
        a = torch.stack((tr, nd), dim=-1)
        # get corresponding log likelihood
        ll = logits.gather(-1, a_idx.unsqueeze(-1))
        assert (ll > -1000).data.all(), "Log_probs are -inf, check sampling procedure!"
        return a, ll, ent

    def set_decode_type(self, decode_type: str) -> None:
        """
        Set the decoding type:
            - 'greedy'
            - 'sampling'

        Args:
            decode_type (str): type of decoding

        """
        self.decode_type = decode_type


class ActionDist(Categorical):
    """
    Adapted categorical distribution that
    ignores -inf masks when calculating entropy.
    """

    def entropy(self) -> torch.Tensor:
        """Ignores -inf in calculation"""
        non_inf_idx = self.logits >= -10000
        p_log_p = self.logits[non_inf_idx] * self.probs[non_inf_idx]
        return -p_log_p.sum(-1)


# ============= #
# ### TEST #### #
# ============= #
def _test():
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS, RPEnv

    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    SMP = 9
    N = 20
    BS = 3
    MAX_CON = 3
    CUDA = False
    SEED = 123

    device = torch.device("cuda" if CUDA else "cpu")
    torch.manual_seed(SEED-1)

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(SEED)
    data = ds.sample(sample_size=SMP, graph_size=N)

    dl = DataLoader(
        data,
        batch_size=BS,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(debug=True, device=device, max_concurrent_vehicles=MAX_CON, k_nbh_frac=0.4)
    env.seed(SEED+1)

    model = Policy(
        observation_space=env.OBSERVATION_SPACE,
        embedding_dim=128,
    )

    for batch in dl:
        model.reset_static()
        env.load_data(batch)
        obs = env.reset()
        done = False
        i = 0

        while not done:
            #print(i)
            action, log_likelihood, entropy = model(obs, recompute_static=(i==0))
            obs, rew, done, info = env.step(action)
            i += 1

        print(info)
