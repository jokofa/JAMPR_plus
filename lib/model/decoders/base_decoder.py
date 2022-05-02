#
from abc import abstractmethod
from torch import Tensor, BoolTensor
import torch.nn as nn


class BaseDecoder(nn.Module):
    """Abstract decoder model."""
    def __init__(self,
                 query_emb_dim: int,
                 action_emb_dim: int,
                 hidden_dim: int = 128,
                 **kwargs):
        """

        Args:
            query_emb_dim: dimension of query embedding
            action_emb_dim: dimension of action embedding
            hidden_dim: dimension of hidden layers
        """
        super(BaseDecoder, self).__init__()
        self.query_emb_dim = query_emb_dim
        self.action_emb_dim = action_emb_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def create_layers(self, **kwargs):
        """Create the specific model layers."""
        raise NotImplementedError

    def reset_parameters(self):
        pass

    def forward(self,
                query_emb: Tensor,
                action_emb: Tensor,
                mask: BoolTensor,
                **kwargs) -> Tensor:
        """
        Model specific implementation of forward pass.

        Args:
            query_emb: (BS, query_emb_dim) query embedding
            action_emb: (BS, num_a, action_emb_dim) action embedding
            mask: (BS, num_a) mask indicating infeasible action indices over action set

        Returns:
            logits: logits over action set

        """
        raise NotImplementedError
