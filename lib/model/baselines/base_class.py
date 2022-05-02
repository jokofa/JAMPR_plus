#
from typing import Iterable, Dict, OrderedDict, Union, List
from torch import Tensor
from torch.utils.data import Dataset

from lib.routing import RPInstance, RPDataset


class BaselineDataset(Dataset):
    """Wrapper to conveniently provide data and baseline values."""
    def __init__(self, dataset: RPDataset, baseline: List):
        super(BaselineDataset, self).__init__()
        assert (len(dataset) == len(baseline))
        self.dataset = dataset
        self.baseline = baseline

    def __getitem__(self, item: int):
        return (
            self.dataset[item],
            self.baseline[item]
        )

    def __len__(self):
        return len(self.dataset)


class Baseline:
    """Abstract baseline class"""

    def wrap_dataset(self, dataset: RPDataset) -> Union[RPDataset, BaselineDataset]:
        """Wrap a dataset with baseline values, if they can be pre-computed."""
        return dataset

    def unwrap_batch(self, batch: List):
        """Unwrap the wrapped dataset."""
        return batch, None

    def eval(self, batch: List[RPInstance], cost: Union[float, Tensor]):
        """Evaluate baseline model on batch."""
        raise NotImplementedError()

    def get_learnable_parameters(self) -> Iterable:
        """Return trainable parameters if any."""
        return []

    def epoch_callback(self, model, epoch: int):
        """Callback to do necessary processing and updates
        at end of each training epoch."""
        pass

    def state_dict(self) -> Union[Dict, OrderedDict]:
        """Pytorch style state_dict"""
        return {}

    def load_state_dict(self, state_dict: Union[Dict, OrderedDict]):
        """Load previously saved state_dict."""
        pass




