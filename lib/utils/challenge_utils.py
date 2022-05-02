#
from typing import Union
import numpy as np
import torch


def dimacs_challenge_dist_fn_np(i: Union[np.ndarray, float],
                                j: Union[np.ndarray, float],
                                scale: int = 100,
                                ) -> np.ndarray:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j

    coords*100 since they were normalized to [0, 1]
    """
    return np.floor(10*np.sqrt(((scale*(i - j))**2).sum(axis=-1)))/10


@torch.jit.script
def dimacs_challenge_dist_fn(i: torch.Tensor, j: torch.Tensor, scale: int = 100,) -> torch.Tensor:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j

    coords*100 since they were normalized to [0, 1]
    """
    return torch.floor(10*torch.sqrt(((scale*(i - j))**2).sum(dim=-1)))/10


# ============= #
# ### TEST #### #
# ============= #
def _test():
    rnd = np.random.default_rng(1)
    np_coords = rnd.uniform(0, 1, size=20).reshape(-1, 2)
    np_dists = dimacs_challenge_dist_fn_np(np_coords[1:], np_coords[0])
    pt_coords = torch.from_numpy(np_coords)
    pt_dists = dimacs_challenge_dist_fn(pt_coords[1:], pt_coords[0])
    assert np.all(np_dists == pt_dists.cpu().numpy())
