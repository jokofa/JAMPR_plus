#
import os
import logging
import time
from warnings import warn
from typing import Optional, Dict, Union, OrderedDict
from omegaconf import DictConfig, OmegaConf as oc

import random
import numpy as np
import torch
import torch.optim as optim

from lib.routing import RPEnv, RPDataset
from lib.model import Policy
from lib.model.baselines import *
from lib.model.training import train, validate
from lib.utils import get_lambda_decay
from lib.utils.runner_utils import (
    CheckpointCallback,
    MonitorCallback,
    update_path,
    remove_dir_tree
)

logger = logging.getLogger(__name__)


#
class Runner:
    """
    Wraps all setup, training and evaluation functionality
    of the respective experiment run configured by cfg.
    """
    def __init__(self, cfg: DictConfig):

        # fix path aliases changed by hydra
        self.cfg = update_path(cfg)

        # debug level
        if (self.cfg.run_type == "debug") or self.cfg.debug_lvl > 0:
            self.debug = max(self.cfg.debug_lvl, 1)
        else:
            self.debug = 0
        if self.debug > 1:
            torch.autograd.set_detect_anomaly(True)

        # set device
        if torch.cuda.is_available() and not cfg.cuda:
            warn(f"Cuda GPU is available but not used! Specify <cuda=True> in config file.")
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        # raise error on strange CUDA warnings which are not propagated
        if (self.cfg.run_type == "train") and cfg.cuda and not torch.cuda.is_available():
            e = "..."
            try:
                torch.zeros(10, device=torch.device("cuda"))
            except Exception as e:
                pass
            raise RuntimeError(f"specified GPU training run but running on CPU! {e}")

    def setup(self):
        """set up all entities."""
        self._dir_setup()
        self._build_env()
        self._build_dataset()
        self._build_policy()
        self._build_baseline()
        self._build_optimizer()
        self._build_callbacks()
        self.seed_all(self.cfg.global_seed)

    def _dir_setup(self):
        """Set up directories for logging, checkpoints, etc."""
        self._cwd = os.getcwd()
        # tb logging dir
        self.cfg.tb_log_path = os.path.join(self._cwd, self.cfg.tb_log_path)
        # checkpoint save dir
        self.cfg.checkpoint_save_path = os.path.join(self._cwd, self.cfg.checkpoint_save_path)
        # val log dir
        self.cfg.val_log_path = os.path.join(self._cwd, self.cfg.val_log_path)
        os.makedirs(self.cfg.val_log_path, exist_ok=True)

    def _build_env(self):
        """Create and wrap the problem environments."""
        env_cfg = self.cfg.env_cfg.copy()

        self.env = RPEnv(
            device=self.device,
            debug=self.debug,
            **env_cfg
        )
        self.env.seed(self.cfg.global_seed)

        # overwrite cfg for validation env
        val_env_cfg = self.cfg.env_cfg.copy()
        val_env_cfg.update(self.cfg.get('val_env_cfg', {}))
        render = self.cfg.get('render_val', False)
        if render:
            val_env_cfg['enable_render'] = True
            val_env_cfg['plot_save_dir'] = self.cfg.val_log_path
        val_env_cfg['num_samples'] = 1      # for evaluation on val set use just 1 trajectory

        self._val_env_cfg = val_env_cfg.copy()
        self.val_env = RPEnv(
            device=self.device,
            debug=self.debug,
            **val_env_cfg
        )
        self.val_env.seed(self.cfg.global_seed+1)

    def _build_dataset(self):
        """Create training dataset sampler and load validation data."""
        train_ds_cfg = self.cfg.train_ds_cfg.copy()
        self.train_ds = RPDataset(**train_ds_cfg)
        val_ds_cfg = self.cfg.val_ds_cfg.copy()
        self.val_ds = RPDataset(**val_ds_cfg)

    def _build_policy(self, state_dict: Optional[OrderedDict] = None):
        """Initialize the policy model."""
        policy_cfg = self.cfg.policy_cfg.copy()
        self.policy = Policy(
            observation_space=self.env.OBSERVATION_SPACE,
            device=self.device,
            **policy_cfg
        )
        if state_dict is not None and 'policy' in state_dict.keys():
            self.policy.load_state_dict(state_dict['policy'])
        logger.info(self.policy)

    def _build_baseline(self, state_dict: Optional[OrderedDict] = None):
        """Initialize the policy gradient baseline."""
        baseline_cfg = self.cfg.baseline_cfg.copy()
        bl_type = baseline_cfg.baseline_type
        bl_args = baseline_cfg.baseline_args

        if bl_type == 'exponential':
            baseline = ExponentialBaseline(**bl_args)
        elif bl_type == 'critic':
            raise NotImplementedError
            #critic_cfg.update(cfg.get('critic_cfg', {}))
            #critic = CriticModel(**critic_cfg)
            #critic = critic.to(device)
            #baseline = CriticBaseline(critic)
        elif bl_type == 'rollout':
            # create baseline val dataset and env
            # (same configuration as for training -
            # i.e. sampling on demand and no rendering)
            env_cfg = self.cfg.env_cfg.copy()
            env = RPEnv(
                device=self.device,
                debug=self.debug,
                **env_cfg
            )
            env.seed(self.cfg.global_seed)
            ds_cfg = self.cfg.train_ds_cfg.copy()
            ds = RPDataset(**ds_cfg)
            baseline = RolloutBaseline(
                dataset=ds,
                env=env,
                policy=self.policy,
                verbose=self.debug > 0,
                **bl_args
            )
        elif bl_type.upper() == "POMO":
            baseline = POMOBaseline(num_samples=self.env.num_samples, **bl_args)
            baseline_cfg.warmup = False     # no warmup for POMO baseline
        else:
            if bl_type is not None and bl_type.lower() != "none":
                raise RuntimeError(f"Unknown baseline: '{bl_type}'")
            # else use dummy baseline
            baseline = NoBaseline()

        # enable warmup procedure
        if baseline_cfg.warmup:
            baseline = WarmupBaseline(baseline, **baseline_cfg.warmup_args)

        # Load baseline checkpoint (make sure script is called with same type of baseline!)
        if state_dict is not None and 'baseline' in state_dict.keys():
            baseline.load_state_dict(state_dict['baseline'])

        self.baseline = baseline

    def _build_optimizer(self, state_dict: Optional[OrderedDict] = None):
        """Initialize optimizer and optional learning rate schedule."""
        optimizer_cfg = self.cfg.optimizer_cfg.copy()

        # create optimizer
        bl_param = self.baseline.get_learnable_parameters()
        optimizer = optim.Adam(
            [{'params': self.policy.parameters(), 'lr': optimizer_cfg['lr_policy']}]
            + (
                [{'params': bl_param, 'lr': optimizer_cfg.get('lr_critic', 1e-3)}]
                if len(bl_param) > 0 else []
            )
        )
        # Load optimizer state
        last_epoch = -1
        if state_dict is not None and 'optimizer' in state_dict.keys():
            optimizer.load_state_dict(state_dict['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            last_epoch = state_dict.get('epoch', -1)
        self.optimizer = optimizer

        # create learning rate scheduler
        scheduler_cfg = self.cfg.scheduler_cfg.copy()
        if scheduler_cfg.schedule_type is not None:
            decay_fn = get_lambda_decay(**scheduler_cfg)
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay_fn, last_epoch=last_epoch)
        else:
            self.lr_scheduler = None

    def _build_callbacks(self):
        """Create necessary callbacks."""
        self.callbacks = {}
        self.callbacks["ckpt_cb"] = CheckpointCallback(
            runner=self,
            save_dir=self.cfg.checkpoint_save_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.checkpoint_cfg
        )
        self.callbacks["monitor"] = MonitorCallback(
            tb_log_path=self.cfg.tb_log_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.monitor_cfg
        )

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.seed(seed)
        self.val_env.seed(seed)
        self.train_ds.seed(seed)
        self.val_ds.seed(seed)

    def train(self, **kwargs):
        """Train the specified model."""
        trainer_cfg = self.cfg.trainer_cfg.copy()
        logger.info(f"start training...")
        result = train(
            cfg=trainer_cfg,
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            train_env=self.env,
            val_env=self.val_env,
            policy=self.policy,
            optimizer=self.optimizer,
            baseline=self.baseline,
            lr_scheduler=self.lr_scheduler,
            verbose=self.debug,
            render_val=self.cfg.render_val,
            **self.callbacks,
            **kwargs
        )
        logger.info(f"training finished.")
        logger.info(result)

    def test(self, test_cfg: Optional[Union[DictConfig, Dict]] = None, **kwargs):
        """Test (evaluate) the provided trained model."""

        # load checkpoint
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        logger.info(f"loading model checkpoint: {ckpt_pth}")
        state_dict = torch.load(ckpt_pth, map_location=self.device)

        # get tester_cfg from current (test) cfg file
        tester_cfg = self.cfg.tester_cfg.copy()
        tb_log_path = os.path.join(os.getcwd(), self.cfg.tb_log_path)
        # buffer current ds cfg with new path
        test_ds_cfg = self.cfg.test_ds_cfg.copy()

        # get checkpoint cfg and update
        self.cfg.update(state_dict["cfg"])
        # update cfg with additionally provided args
        if test_cfg is not None:
            test_cfg = oc.to_container(test_cfg, resolve=True) if isinstance(test_cfg, DictConfig) else test_cfg
            tester_cfg.update(test_cfg.get('tester_cfg', {}))
            self.cfg.update(test_cfg)

        sampling = tester_cfg.get("sampling", False)
        num_samples = tester_cfg.get("num_samples", 1)

        # create test env
        test_env_cfg = self.cfg.env_cfg.copy()
        test_env_cfg.update(tester_cfg.get('test_env_cfg', {}))
        if sampling or test_env_cfg.get("pomo", False):
            test_env_cfg["num_samples"] = max(num_samples, test_env_cfg.get("num_samples", 1))

        render = tester_cfg.get('render', False)
        if render:
            test_env_cfg['enable_render'] = True
            test_env_cfg['plot_save_dir'] = self.cfg.test_log_path

        self.env = RPEnv(
            device=self.device,
            debug=self.debug,
            **test_env_cfg
        )
        self.env.seed(self.cfg.global_seed+2)

        # create dataset wrapper
        test_ds = RPDataset(**test_ds_cfg)
        test_ds.load_ds()  # loads dataset from path

        # load model weights
        self._build_policy(state_dict)

        # create callback
        monitor = MonitorCallback(
            tb_log_path=tb_log_path,
            metric_key=self.cfg.eval_metric_key,
            **self.cfg.monitor_cfg
        )

        # run test inference
        logger.info("running test inference...")
        t_start = time.time()
        result = validate(
            dataset=test_ds,
            env=self.env,
            policy=self.policy,
            batch_size=tester_cfg['test_batch_size'],
            num_workers=0,
            disable_progress_bar=tester_cfg['disable_progress_bar'],
            render=render,
            sampling=sampling,
        )
        t_total = time.time() - t_start
        logger.info(f"finished after {t_total}s.")

        monitor.log_eval_data(result, 0, mode="test")
        logger.info(result)

    def resume(self, **kwargs):
        """Resume training from checkpoint."""
        ckpt_pth = self.cfg.get('checkpoint_load_path')
        assert ckpt_pth is not None
        state_dict = torch.load(ckpt_pth, map_location=self.device)
        self.load_state_dict(state_dict)

        # remove the unnecessary new directory hydra creates
        new_hydra_dir = os.getcwd()
        if "resume" in new_hydra_dir:
            remove_dir_tree("resume", pth=new_hydra_dir)

        logger.info(f"resuming training from: {ckpt_pth}")
        self.train(resume=True, **kwargs)

    def state_dict(self, epoch: int = 0, **kwargs) -> Dict:
        """Save states of all experiment components
        in PyTorch like state dictionary."""
        return {
            "cfg": self.cfg.copy(),
            "policy": self.policy.state_dict(**kwargs),
            "baseline": self.baseline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": torch.get_rng_state().cpu(),
            "epoch": epoch,
        }

    def load_state_dict(self, state_dict: OrderedDict):
        """
        Load a previously saved state_dict and
        reinitialize all required components.

        Examples:
            state_dict = torch.load(PATH)
            experiment.load_state_dict(state_dict)
        """
        self.cfg.update(state_dict["cfg"])
        self._dir_setup()
        self._build_env()
        self._build_dataset()
        self._build_policy(state_dict)
        self._build_baseline(state_dict)
        self._build_optimizer(state_dict)
        self._build_callbacks()
        torch.set_rng_state(state_dict["rng"].cpu())

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug']:
            self.setup()
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type == 'test':
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'test', 'debug']")
