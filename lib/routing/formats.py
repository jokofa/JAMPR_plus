#
from typing import NamedTuple, Union, Any, List
from torch import Tensor, LongTensor, BoolTensor
import numpy as np
import torch

__all__ = [
    "RPInstance",
    "RPObs",
]


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"


class RPInstance(NamedTuple):
    """Typed routing problem instance wrapper."""
    coords: Union[np.ndarray, torch.Tensor]
    demands: Union[np.ndarray, torch.Tensor]
    tw: Union[np.ndarray, torch.Tensor]
    service_time: Union[np.ndarray, torch.Tensor, float]
    graph_size: int
    org_service_horizon: Union[float, int]
    max_vehicle_number: int
    vehicle_capacity: float = 1.0
    service_horizon: float = 1.0
    depot_idx: List = [0]
    type: Union[int, str] = ""
    tw_frac: Union[float, str] = ""

    @property
    def node_features(self):
        return np.concatenate((self.coords, self.demands, self.tw), axis=-1)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val


class RPObs(NamedTuple):
    """Named and typed tuple of RP observations."""
    batch_size: int
    node_features: Tensor
    node_nbh_edges: LongTensor
    node_nbh_weights: Tensor
    tour_plan: LongTensor
    tour_features: Tensor
    tour_edges: LongTensor
    tour_weights: Tensor
    nbh: LongTensor
    nbh_mask: BoolTensor
