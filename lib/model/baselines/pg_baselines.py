#
import copy
import logging
from typing import Dict, Optional, OrderedDict, List, Union
from torch import Tensor

from scipy.stats import ttest_rel
import torch
import torch.nn.functional as F

from lib.model.baselines.base_class import Baseline, BaselineDataset
from lib.routing import RPInstance, RPDataset, RPEnv
from lib.model.policy import Policy
from lib.model.training import rollout, eval_episode

__all__ = [
    "NoBaseline",
    "ExponentialBaseline",
    "WarmupBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "POMOBaseline",
]

logger = logging.getLogger(__name__)


class NoBaseline(Baseline):
    """Dummy baseline doing nothing."""
    def eval(self, batch: List[RPInstance], cost: Union[float, Tensor]):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    """Exponential moving average baseline."""
    def __init__(self, beta: float, **kwargs):
        super(Baseline, self).__init__()
        self.beta = beta
        self.v = None

    @torch.no_grad()
    def eval(self, batch: List[RPInstance], cost: Tensor):
        if self.v is None:
            v = cost.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * cost.mean()

        self.v = v.detach()
        return self.v, 0  # No loss

    def state_dict(self):
        return {'v': self.v}

    def load_state_dict(self, state_dict: Union[Dict, OrderedDict]):
        self.v = state_dict['v']


class WarmupBaseline(Baseline):
    """Wrapper implementing exponential warmup phase
    for rollout and critic baselines."""
    def __init__(self,
                 baseline: Baseline,
                 n_epochs: int = 1,
                 warmup_exp_beta: float = 0.8,
                 verbose: bool = False,
                 **kwargs):
        super(Baseline, self).__init__()

        self.baseline = baseline
        self.n_epochs = n_epochs
        self.verbose = verbose
        assert n_epochs > 0, "n_epochs must be positive."
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta, **kwargs)
        self.alpha = 0

    def wrap_dataset(self, dataset: RPDataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch: List):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    @torch.no_grad()
    def eval(self, batch: List[RPInstance], cost: Union[float, Tensor]):
        if self.alpha >= 1:
            return self.baseline.eval(batch, cost)
        elif self.alpha == 0:
            return self.warmup_baseline.eval(batch, cost)
        else:
            # Return convex combination of baselines
            v, l = self.baseline.eval(batch, cost)
            vw, lw = self.warmup_baseline.eval(batch, cost)
            return (
                self.alpha * v + (1 - self.alpha) * vw,
                self.alpha * l + (1 - self.alpha * (lw if lw is not None else 0))
            )

    def epoch_callback(self, policy: Policy, epoch: int):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(policy, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if self.verbose and epoch < self.n_epochs:
            logger.info(f"Set warmup alpha = {self.alpha}")

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class RolloutBaseline(Baseline):
    """Greedy rollout baseline.

    Args:
        dataset: baseline val dataset
        env: baseline val env
        policy: policy model
        sample_size: size of baseline val dataset
        graph_size: size of graphs in baseline val dataset
        batch_size: batch size for baseline evaluations
        check_significance: flag to use ttest to check significance of baseline improvement
        alpha: confidence level for p-value
        resample_interval: resample the baseline validation data every 'resample_interval' epochs
                            to prevent over-fitting (disabled for interval=0)
        num_workers: number of workers to load data
        verbose: flag to log additional info

    """
    def __init__(self,
                 dataset: RPDataset,
                 env: RPEnv,
                 policy: Policy,
                 sample_size: int,
                 graph_size: int,
                 batch_size: int,
                 check_significance: bool = True,
                 alpha: float = 0.05,
                 resample_interval: int = 0,
                 num_workers: int = 4,
                 verbose: bool = False,
                 **kwargs):
        super(Baseline, self).__init__()

        self.dataset = dataset
        self.env = env
        self.sample_size = sample_size
        self.graph_size = graph_size
        self.batch_size = batch_size
        self.check_significance = check_significance
        assert alpha > 0
        self.alpha = alpha
        self.resample_interval = resample_interval
        self.num_workers = num_workers
        self.verbose = verbose

        self.policy = None
        self.bl_vals = None
        self.bl_epoch = 0
        self._cpu = torch.device("cpu")
        self._device = None
        self._last_sample_epoch = 0

        self._update_model(policy, 0)
        self._clear()

    def _update_model(self, policy: Policy, epoch: int):
        self.bl_epoch = epoch
        self._device = policy.device
        self.policy = copy.deepcopy(policy)

        resample = self.check_significance and (
            self._last_sample_epoch == 0 or
            (self.resample_interval > 0 and epoch - self._last_sample_epoch >= self.resample_interval)
        )
        if resample:
            # (re)sample val dataset
            self.dataset = self.dataset.sample(sample_size=self.sample_size,
                                               graph_size=self.graph_size)
            self._last_sample_epoch = epoch

            # (re)evaluate in case of new data
            if self.verbose:
                logger.info("Evaluating baseline model on evaluation dataset")
            self.bl_vals = rollout(
                dataset=self.dataset,
                env=self.env,
                policy=self.policy.to(self._device),    # move to GPU for fast process
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                disable_progress_bar=True,  # always disable
            )[0].cpu()

    def _clear(self):
        # move model to CPU and clear env cache, since is not needed the whole remaining epoch
        self.policy.to(device=self._cpu)
        self.env.clear_cache()

    def wrap_dataset(self, dataset: RPDataset) -> BaselineDataset:
        if self.verbose:
            logger.info(f"Precomputing baseline on dataset ({len(dataset)} instances)...")
        bl = rollout(
            dataset=dataset,
            env=self.env,
            policy=self.policy.to(self._device),    # move to GPU for fast pre-computing
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            disable_progress_bar=True,  # always disable
        )[0].cpu()
        self._clear()
        return BaselineDataset(dataset, bl)

    def unwrap_batch(self, batch):
        return (
            [b[0] for b in batch],
            torch.stack([b[1] for b in batch], dim=0).to(self._device)
        )

    @torch.no_grad()
    def eval(self, batch: List[RPInstance], cost: Union[float, Tensor]):
        # single batch so we do not need rollout function
        cost, _ = eval_episode(batch, self.env, self.policy)
        return cost, 0   # no loss

    def epoch_callback(self, policy: Policy, epoch: int):
        """
        Challenge the current baseline with the new model and
        replace the baseline model if the new model is significantly better.

        Args:
            policy: policy to challenge the baseline by
            epoch: current epoch

        Returns:

        """
        if not self.check_significance:
            self._update_model(policy, epoch)
        else:
            if self.verbose:
                logger.info("Evaluating candidate model on baseline val dataset")
            candidate_vals = rollout(
                dataset=self.dataset,
                env=self.env,
                policy=policy,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                disable_progress_bar=True,  # always disable
            )[0].cpu()

            candidate_mean = candidate_vals.mean()
            cur_mean = self.bl_vals.mean()
            diff = candidate_mean - cur_mean

            p_val_str = ""
            if diff < 0:
                # Calc p value
                t, p = ttest_rel(candidate_vals.numpy(), self.bl_vals)
                assert t < 0, "T-statistic should be negative"
                p_val = p / 2  # one-sided
                p_val_str = f", p-value: {p_val: .6f}"

                if p_val < self.alpha:
                    if self.verbose:
                        logger.info('Update baseline')
                    self.bl_vals = candidate_vals
                    self._update_model(policy, epoch)

            if self.verbose:
                logger.info(f"Epoch {epoch} / Baseline epoch {self.bl_epoch}"
                            f"\n    candidate mean {candidate_mean: .6f}, "
                            f"\n    baseline  mean {cur_mean: .6f}, "
                            f"\n    difference {diff: .6f}{p_val_str}")

    def state_dict(self):
        return {
            'bl_policy': self.policy,
            'bl_dataset': self.dataset,
            'bl_epoch': self.bl_epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        policy_ = copy.deepcopy(self.policy)
        policy_.load_state_dict(state_dict['bl_policy'].state_dict())
        self._update_model(policy_, state_dict['bl_epoch'])


class CriticBaseline(Baseline):

    def __init__(self, critic: torch.nn.Module):
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, batch: List[RPInstance], cost: Union[float, Tensor]):
        v = self.critic(batch)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, cost.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, policy: Policy, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class POMOBaseline(Baseline):
    """Uses the average cost over the POMO rollouts as baseline."""
    def __init__(self, num_samples: int, **kwargs):
        super(Baseline, self).__init__()
        self.num_samples = num_samples

    @torch.no_grad()
    def eval(self, batch: List[RPInstance], cost: Tensor):
        # cost has shape (BS*num_samples, ) where num samples is the number of POMO rollout trajectories
        bs = len(batch)
        v = cost.view(bs, self.num_samples).mean(dim=-1)[:, None].expand(-1, self.num_samples).reshape(-1)
        return v.detach(), 0  # No loss

