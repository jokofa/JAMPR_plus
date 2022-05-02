#
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor

from lib.model.decoders.base_decoder import BaseDecoder


class AttnDecoder(BaseDecoder):
    """
    Attention decoder model.

    adapted from:
        Kool, W., Van Hoof, H., & Welling, M. (2018).
        Attention, learn to solve routing problems!.
        arXiv preprint arXiv:1803.08475.

    """
    def __init__(self,
                 query_emb_dim: int,
                 action_emb_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 clip_tanh: Union[int, float] = 10.,
                 bias: bool = False,
                 **kwargs):
        super(AttnDecoder, self).__init__(
            query_emb_dim,
            action_emb_dim,
            hidden_dim
        )

        self.num_heads = num_heads
        self.clip_tanh = clip_tanh
        self.bias = bias

        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, "<hidden_dim> must be divisible by <num_heads>!"
        self.head_dim = head_dim

        # scaling factors for scaled product attention
        self.u_norm = (float(head_dim) ** -0.5)
        self.nc_norm = (float(hidden_dim) ** -0.5)

        self.ctxt_proj, self.set_proj, self.out_proj = None, None, None
        self.create_layers(**kwargs)

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        self.ctxt_proj = nn.Linear(self.query_emb_dim, self.hidden_dim, bias=self.bias)
        # set_proj -> glimpse_key, glimpse_val, logit_key
        self.set_proj = nn.Linear(self.action_emb_dim, 3 * self.hidden_dim, bias=self.bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)

    def reset_parameters(self):
        self.ctxt_proj.reset_parameters()
        self.set_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self,
                query_emb: Tensor,
                action_emb: Tensor,
                mask: BoolTensor,
                **kwargs):

        # calculate projections and create heads
        ctxt = self._make_heads(self.ctxt_proj(query_emb[:, None, :]))
        # split projection of size 3x hidden_dim: glimpses -> (n_heads, BS*M, hdim)
        glimpse_key, glimpse_val, logit_key = self._split_set(self.set_proj(action_emb))

        # compatibility (scoring) --> (n_heads, BS, 1, K)
        x = torch.matmul(ctxt, glimpse_key.transpose(-2, -1)) * self.u_norm
        # mask compatibility
        if mask is not None:
            x[mask[None, :, None, :].expand_as(x)] = float('-inf')

        # compute attention heads --> (n_heads, BS, K, 1, head_dim)
        x = torch.matmul(F.softmax(x, dim=-1), glimpse_val)

        # calculate projection for updated context embedding (BS, 1, hdim)
        x = self.out_proj(
            x.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.hidden_dim)
        )

        # compute logits --> (BS, K)
        x = torch.matmul(x, logit_key.transpose(-2, -1)).squeeze(-2) * self.nc_norm

        # tanh clipping/saturation
        if self.clip_tanh:
            x = torch.tanh(x) * self.clip_tanh

        # apply mask
        if mask is not None:
            x[mask] = float('-inf')

        return F.log_softmax(x, dim=-1)  # logits

    def _make_heads(self, x: Tensor):
        """Makes attention heads for the provided glimpses (BS, N, emb_dim)"""
        return (
            x.contiguous()
            .view(x.size(0), x.size(1), self.num_heads, -1)  # emb_dim --> head_dim * n_heads
            .permute(2, 0, 1, 3)  # (n_heads, BS, N, head_dim)
        )

    def _split_set(self, x):
        """Split projected tensor into required components."""
        glimpse_key, glimpse_val, logit_key = x.chunk(3, dim=-1)
        return self._make_heads(glimpse_key), self._make_heads(glimpse_val), logit_key.contiguous()


# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 5,
        n: int = 10,
        k: int = 3,
        cuda=False,
        seed=1
):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    QDIM = 64
    ADIM = 32
    num_a = (n//2)*k

    q_emb = torch.randn(bs, QDIM).to(device)
    a_emb = torch.randn(bs, num_a, ADIM).to(device)
    mask = torch.randint(0, 2, (bs, num_a)).to(dtype=torch.bool, device=device)

    dec = AttnDecoder(QDIM, ADIM).to(device)
    logits = dec(q_emb, a_emb, mask)
    assert logits.size() == torch.empty((bs, num_a)).size()
    return True
