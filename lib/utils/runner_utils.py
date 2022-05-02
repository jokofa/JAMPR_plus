#
import os
import shutil
import logging
from warnings import warn
from typing import Dict, Optional, Tuple, Callable, Union, Any
from omegaconf import DictConfig

import hydra
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

LOG_DATA_TYPE = Dict[str, Union[int, float, np.number, np.ndarray]]


class CheckpointCallback:
    """
    Callback to manage checkpoints.
    Is called by MonitorCallback.

    Args:
        runner: runner object
        save_dir: directory for saved checkpoints
        fname: file name prefix (default='model')
        metric_key: key of metric (default='cost')
        compare_mode: mode of eval metric comparison, (rew -> 'max', cost -> 'min')
        top_k: number of checkpoints to keep
    """
    FILE_EXTS = ".ckpt"

    def __init__(self,
                 runner,
                 save_dir: str,
                 fname: str = "model",
                 metric_key: str = "cost",
                 compare_mode: str = "min",
                 top_k: int = 1,
                 save_last: bool = True,
                 **kwargs):
        self.runner = runner
        self.save_dir = save_dir
        self.prefix = fname
        self.metric_key = metric_key
        self.compare_mode = compare_mode
        self.top_k = top_k
        self.save_last = save_last

        os.makedirs(self.save_dir, exist_ok=True)
        v = -float("inf") if self.compare_mode == "max" else float("inf")
        self.top_k_checkpoints = [{'eval_metric': v, 'pth': None} for _ in range(top_k)]

    def __call__(self, epoch: int, eval_metric: Union[float, Any], is_last: bool = False):
        """Called to compare existing checkpoints and save top-k best models."""
        if is_last:
            if self.save_last:
                # save final checkpoint no matter how good
                m = f"{eval_metric: .6f}".lstrip()
                fname = f"ep{epoch}(last)_{self.prefix}_{self.metric_key}={m}{self.FILE_EXTS}"
                add_pth = os.path.join(self.save_dir, fname)
                # save
                logger.info(f"Saving last checkpoint to: {add_pth}")
                torch.save(self.runner.state_dict(epoch=epoch), add_pth)

        else:
            # check metric
            if eval_metric is None:
                warn(f"Eval metric is None. No checkpoint saved.")
            else:
                is_better, idx = self._compare_metric(eval_metric)
                if is_better:
                    # delete worst checkpoint
                    del_pth = self.top_k_checkpoints.pop(-1)['pth']
                    if del_pth is not None and os.path.exists(del_pth):
                        os.remove(del_pth)
                    # add new checkpoint
                    m = f"{eval_metric: .6f}".lstrip()
                    fname = f"ep{epoch}_{self.prefix}_{str(self.metric_key)}={m}{self.FILE_EXTS}"
                    add_pth = os.path.join(self.save_dir, fname)
                    self.top_k_checkpoints.insert(idx, {'eval_metric': eval_metric, 'pth': add_pth})
                    # save
                    logger.info(f"Saving new checkpoint to: {add_pth}")
                    torch.save(self.runner.state_dict(epoch=epoch), add_pth)

    def _compare_metric(self, eval_metric: float) -> Tuple[bool, int]:
        cur_best = np.array([cp['eval_metric'] for cp in self.top_k_checkpoints])
        if self.compare_mode == "max":
            check = cur_best < eval_metric
        elif self.compare_mode == "min":
            check = cur_best > eval_metric
        else:
            raise ValueError(f"unknown compare mode: '{self.compare_mode}'")
        is_better = np.any(check)
        # since list is ordered by metric, first nonzero position is target position for insertion
        idx = np.nonzero(check)[0][0] if is_better else None
        return is_better, idx


class MonitorCallback:
    """
    TB logger to monitor eval metrics
    for calling the checkpoint callback.

    adapted from Tianshou logger:
    https://github.com/thu-ml/tianshou/blob/master/tianshou/utils/logger/tensorboard.py

    Args:
        tb_log_path: directory to save tensorboard events file
        metric_key: key of eval metric to compare
        train_interval: log training results every train_interval episodes
        eval_interval: log validation results every eval_interval episodes
        save_interval: save model and results every save_interval epochs
    """
    def __init__(self,
                 tb_log_path: str,
                 metric_key: str = "cost",
                 train_interval: int = 100,
                 eval_interval: int = 100,
                 save_interval: int = 1,
                 ):
        self.tb_log_path = tb_log_path
        self.metric_key = metric_key
        self.train_interval = train_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # create TB summary writer
        self.writer = SummaryWriter(tb_log_path)

        self.last_log_train_step = -1
        self.last_log_eval_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1

        self.eval_metric = None
        os.makedirs(self.tb_log_path, exist_ok=True)

    def write(self, step: int, data: LOG_DATA_TYPE) -> None:
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step=step)
        self.writer.flush()

    def log_train_data(self, result: dict, step: int) -> None:
        if step - self.last_log_train_step >= self.train_interval:
            log_data = {
                "train/n_episodes": step,
                "train/cost": result["cost"],
                "train/num_vehicles_used": result["k_used"],
                "train/pg_loss": result["pg_loss"],
                "train/critic_loss": result["critic_loss"],
                "train/entropy": result["entropy"],
                "train/grad_norm": result["grad_norm"],
            }
            if "lr" in result.keys():
                log_data["lr"] = result["lr"]
            self.write(step, log_data)
            self.last_log_train_step = step
    
    def log_eval_data(self, result: dict, step: int, mode: str = "test") -> None:
        if step - self.last_log_eval_step >= self.eval_interval:
            log_data = {
                f"{mode}/n_episodes": step,
                f"{mode}/cost": result["cost"],
                f"{mode}/num_vehicles_used": result["k_used"],
                f"{mode}/cost_std": result["cost_std"],
                f"{mode}/num_vehicles_used_std": result["k_used_std"],
                f"{mode}/num_vehicles_used_max": result["k_used_max"],
            }
            self.write(step, log_data)
            self.last_log_eval_step = step

        self.eval_metric = result.get(self.metric_key, None)

    def save_data(
            self,
            epoch: int,
            env_step: int,
            gradient_step: int,
            checkpoint_cb: Optional[Callable] = None,
            is_last: bool = False,
    ) -> None:
        if checkpoint_cb and epoch - self.last_save_step >= self.save_interval or is_last:
            self.last_save_step = epoch
            # call CheckpointCallback
            checkpoint_cb(epoch, self.eval_metric, is_last)
            # write additional logs
            self.write(epoch, {"save/epoch": epoch})
            self.write(env_step, {"save/env_step": env_step})
            self.write(gradient_step, {"save/gradient_step": gradient_step})

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_eval_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # if has no env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step


def add_append_to_dict(d, k, v):
    if isinstance(d[k], int) or isinstance(d[k], float):
        d[k] += v
    elif isinstance(d[k], np.ndarray):
        d[k] = np.append(d[k], v)
    else:
        raise TypeError(f"Dictionary values need to be of type int, float or numpy.ndarray.")
    return d


def update_path(cfg: DictConfig):
    """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
    cwd = hydra.utils.get_original_cwd()

    if cfg.train_ds_cfg is not None:
        if cfg.train_ds_cfg.data_pth is not None:
            cfg.train_ds_cfg.data_pth = os.path.normpath(
                os.path.join(cwd, cfg.train_ds_cfg.data_pth)
            )
        if cfg.train_ds_cfg.stats_pth is not None:
            cfg.train_ds_cfg.stats_pth = os.path.normpath(
                os.path.join(cwd, cfg.train_ds_cfg.stats_pth)
            )

    if cfg.val_ds_cfg is not None:
        if cfg.val_ds_cfg.data_pth is not None:
            cfg.val_ds_cfg.data_pth = os.path.normpath(
                os.path.join(cwd, cfg.val_ds_cfg.data_pth)
            )
        if cfg.val_ds_cfg.stats_pth is not None:
            cfg.val_ds_cfg.stats_pth = os.path.normpath(
                os.path.join(cwd, cfg.val_ds_cfg.stats_pth)
            )

    if cfg.test_ds_cfg is not None:
        if cfg.test_ds_cfg.data_pth is not None:
            cfg.test_ds_cfg.data_pth = os.path.normpath(
                os.path.join(cwd, cfg.test_ds_cfg.data_pth)
            )
        if cfg.test_ds_cfg.stats_pth is not None:
            cfg.test_ds_cfg.stats_pth = os.path.normpath(
                os.path.join(cwd, cfg.test_ds_cfg.stats_pth)
            )

    if cfg.checkpoint_load_path is not None:
        cfg.checkpoint_load_path = os.path.normpath(
                os.path.join(cwd, cfg.checkpoint_load_path)
            )
    return cfg


def remove_dir_tree(root: str, pth: Optional[str] = None):
    """Remove the full directory tree of the root directory if it exists."""
    if not os.path.isdir(root) and pth is not None:
        # select root directory from path by dir name
        i = pth.index(root)
        root = pth[:i+len(root)]
    if os.path.isdir(root):
        shutil.rmtree(root)
