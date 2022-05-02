#
import os
import sys
from typing import Union, Optional, Tuple, List, Dict
import math
import re
from warnings import warn
import itertools as it
import pickle
import logging

import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
import torch
from torch.utils.data import Dataset

from lib.routing.formats import RPInstance
from lib.utils.challenge_utils import dimacs_challenge_dist_fn_np

__all__ = [
    "RPGenerator", "RPDataset",
    'GROUPS', 'TYPES', 'TW_FRACS',
]
logger = logging.getLogger(__name__)

# Solomon instance naming components
GROUPS = ["r", "c", "rc"]
TYPES = ["1", "2"]
TW_FRACS = [0.25, 0.5, 0.75, 1.0]
MAX_TRY = 100


def format_ds_save_path(directory, args=None, affix=None, fname=''):
    """Format the path for saving datasets"""
    directory = os.path.normpath(os.path.expanduser(directory))

    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    if affix is not None:
        fname = str(affix) + fname
    if fname != '':
        fpath = os.path.join(directory, fname)
    else:
        fpath = directory
    if fpath[-3:] not in ['.pt', 'dat', 'pkl']:
        fpath += '.pt'

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        a = input()
        if a != 'y':
            print('Could not write to file. Terminating program...')
            sys.exit()

    return fpath


class CoordsSampler:
    """Sampler implementing different options to generate coordinates for RPs."""
    def __init__(self,
                 n_components: int = 5,
                 n_dims: int = 2,
                 coords_sampling_dist: str = "uniform",
                 covariance_type: str = "diag",
                 mus: Optional[np.ndarray] = None,
                 sigmas: Optional[np.ndarray] = None,
                 mu_sampling_dist: str = "normal",
                 mu_sampling_params: Tuple = (0, 1),
                 sigma_sampling_dist: str = "uniform",
                 sigma_sampling_params: Tuple = (0.025, 0.05),
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                 verbose: bool = False,
                 ):
        """

        Args:
            n_components: number of mixture components
            n_dims: dimension of sampled features, e.g. 2 for Euclidean coordinates
            coords_sampling_dist: type of distribution to sample coordinates, one of ["uniform"]
            covariance_type: type of covariance matrix, one of ['diag', 'full']
            mus: user provided mean values for mixture components
            sigmas: user provided covariance values for mixture components
            mu_sampling_dist: type of distribution to sample initial mus, one of ['uniform', 'normal']
            mu_sampling_params: parameters for mu sampling distribution
            sigma_sampling_dist: type of distribution to sample initial sigmas, one of ['uniform', 'normal']
            sigma_sampling_params: parameters for sigma sampling distribution
            random_state: seed integer or numpy random (state) generator
            verbose: verbosity flag to print additional info and warnings
        """
        self.nc = n_components
        self.f = n_dims
        self.coords_sampling_dist = coords_sampling_dist.lower()
        self.covariance_type = covariance_type
        self.mu_sampling_dist = mu_sampling_dist.lower()
        self.mu_sampling_params = mu_sampling_params
        self.sigma_sampling_dist = sigma_sampling_dist.lower()
        self.sigma_sampling_params = sigma_sampling_params
        self.verbose = verbose
        # set random generator
        if random_state is None or isinstance(random_state, int):
            self.rnd = np.random.default_rng(random_state)
        else:
            self.rnd = random_state

        if self.coords_sampling_dist in ["gm", "gaussian_mixture", "unf+gm"]:
            # sample initial mu and sigma if not provided
            if mus is not None:
                assert (
                    (mus.shape[0] == self.nc and mus.shape[1] == self.f) or
                    (mus.shape[0] == self.nc * self.f)
                )
                self.mu = mus.reshape(self.nc * self.f)
            else:
                self.mu = self._sample_mu(mu_sampling_dist.lower(), mu_sampling_params)
            if sigmas is not None:
                assert (
                    (sigmas.shape[0] == self.nc and sigmas.shape[1] == (self.f if covariance_type == "diag" else self.f**2))
                    or (sigmas.shape[0] == (self.nc * self.f if covariance_type == "diag" else self.nc * self.f**2))
                )
                self.sigma = self._create_cov(sigmas, cov_type=covariance_type)
            else:
                covariance_type = covariance_type.lower()
                if covariance_type not in ["diag", "full"]:
                    raise ValueError(f"unknown covariance type: <{covariance_type}>")
                self.sigma = self._sample_sigma(sigma_sampling_dist.lower(), sigma_sampling_params, covariance_type)

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd = np.random.default_rng(seed)

    def resample_gm(self):
        """Resample initial mus and sigmas."""
        self.mu = self._sample_mu(
            self.mu_sampling_dist,
            self.mu_sampling_params
        )
        self.sigma = self._sample_sigma(
            self.sigma_sampling_dist,
            self.sigma_sampling_params,
            self.covariance_type
        )

    def sample(self,
               n: int,
               resample_mixture_components: bool = True,
               **kwargs) -> np.ndarray:
        """
        Args:
            n: number of samples to draw
            resample_mixture_components: flag to resample mu and sigma of all mixture components for each instance

        Returns:
            coords: (n, n_dims)
        """
        # sample depot from a inner circle
        #depot = self._sample_unf_coords(1, **kwargs)
        depot = self._sample_ring(size=1, radius_range=(0, 0.13))
        depot = (depot + 1)/2   # normalize
        depot = np.maximum(np.minimum(depot, 0.7), 0.3)

        if self.coords_sampling_dist == "uniform":
            coords = self._sample_unf_coords(n, **kwargs)
        else:
            if resample_mixture_components:
                self.resample_gm()
            if self.coords_sampling_dist == "unf+gm":
                n_unf = n//2
                n_gm = n-n_unf
                unf_coords = self._sample_unf_coords(n_unf, **kwargs)
                n_per_c = math.ceil(n_gm / self.nc)
                self.mu = self._sample_mu(dist="ring", params=(0.9, 1.2))
                gm_coords = self._sample_gm_coords(n_per_c, n_gm, **kwargs)
                coords = np.vstack((unf_coords, gm_coords))
            else:
                n_per_c = math.ceil(n / self.nc)
                coords = self._sample_gm_coords(n_per_c, n, **kwargs)

        return np.vstack((depot, coords)).astype(np.float32)

    def _sample_mu(self, dist: str, params: Tuple):
        size = self.nc * self.f
        if dist == "uniform":
            return self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            return self._sample_normal(size, params[0], params[1])
        elif dist == "ring":
            return self._sample_ring(self.nc, params).reshape(-1)
        elif dist == "io_ring":
            return self._sample_io_ring(self.nc).reshape(-1)
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")

    def _sample_sigma(self, dist: str, params: Tuple, cov_type: str):
        if cov_type == "full":
            size = self.nc * self.f**2
        else:
            size = self.nc * self.f
        if dist == "uniform":
            x = self._sample_uniform(size, params[0], params[1])
        elif dist == "normal":
            x = np.abs(self._sample_normal(size, params[0], params[1]))
        else:
            raise ValueError(f"unknown sampling distribution: <{dist}>")
        return self._create_cov(x, cov_type=cov_type)

    def _create_cov(self, x, cov_type: str):
        if cov_type == "full":
            # create block diagonal matrix to model covariance only
            # between features of each individual component
            x = x.reshape((self.nc, self.f, self.f))
            return block_diag(*x.tolist())
        else:
            return np.diag(x.reshape(-1))

    def _sample_uniform(self,
                        size: Union[int, Tuple[int, ...]],
                        low: Union[int, np.ndarray] = 0.0,
                        high: Union[int, np.ndarray] = 1.0):
        return self.rnd.uniform(size=size, low=low, high=high)

    def _sample_normal(self,
                       size: Union[int, Tuple[int, ...]],
                       mu: Union[int, np.ndarray],
                       sigma: Union[int, np.ndarray]):
        return self.rnd.normal(size=size, loc=mu, scale=sigma)

    def _sample_ring(self, size: int, radius_range: Tuple = (0, 1)):
        """inspired by https://stackoverflow.com/a/41912238"""
        #eps = self.rnd.standard_normal(1)[0]
        if size == 1:
            angle = self.rnd.uniform(0, 2*np.pi, size)
            #eps = self.rnd.uniform(0, np.pi, size)
        else:
            angle = np.linspace(0, 2*np.pi, size)
        #angle = np.linspace(0+eps, 2*np.pi+eps, size)
        #angle = rnd.uniform(0, 2*np.pi, size)
        #angle += self.rnd.standard_normal(size)*0.05
        angle += self.rnd.uniform(0, np.pi/3, size)
        d = np.sqrt(self.rnd.uniform(*radius_range, size))
        #d = np.sqrt(rnd.normal(np.mean(radius_range), (radius_range[1]-radius_range[0])/2, size))
        return np.concatenate((
            (d*np.cos(angle))[:, None],
            (d*np.sin(angle))[:, None]
        ), axis=-1)

    def _sample_io_ring(self, size: int):
        """sample an inner and outer ring."""
        # have approx double the number of points in outer ring than inner ring
        num_inner = size//3
        num_outer = size-num_inner
        inner = self._sample_ring(num_inner, (0.01, 0.2))
        outer = self._sample_ring(num_outer, (0.21, 0.5))
        return np.vstack((inner, outer))

    def _sample_unf_coords(self, n: int, **kwargs) -> np.ndarray:
        """Sample coords uniform in [0, 1]."""
        return self.rnd.uniform(size=(n, self.f))

    def _sample_gm_coords(self, n_per_c: int, n: Optional[int] = None, **kwargs) -> np.ndarray:
        """Sample coordinates from k Gaussians."""
        coords = self.rnd.multivariate_normal(
            mean=self.mu,
            cov=self.sigma,
            size=n_per_c,
        ).reshape(-1, self.f)   # (k*n, f)
        if n is not None:
            coords = coords[:n]     # if k % n != 0, some of the components have 1 more sample than others
        # normalize coords in [0, 1]
        return self._normalize_coords(coords)

    @staticmethod
    def _normalize_coords(coords: np.ndarray):
        """Applies joint min-max normalization to x and y coordinates."""
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        max_val = coords.max()  # joint max to preserve relative spatial distances
        coords[:, 0] = coords[:, 0] / max_val
        coords[:, 1] = coords[:, 1] / max_val
        return coords


class InstanceSampler:
    """Sampler class for samplers based on Solomon benchmark data statistics."""
    def __init__(self, cfg: Dict, seed: int = 1):

        # set key-value pairs from Solomon instance stats
        # as InstanceSampler instance attributes
        for k, v in cfg.items():
            setattr(self, k, v)

        # initialize random generator
        self.rnd = np.random.default_rng(seed)
        # set group id (remove instance number from id str)
        self.group_id = re.sub(r'\d+', '', self.id).upper()

        # coords sampler (depends on group id)
        if self.group_id == "R":
            self.coords_sampler = CoordsSampler(
                coords_sampling_dist="uniform",
                random_state=seed+1
            )
        elif self.group_id == "C":
            self.coords_sampler = CoordsSampler(
                n_components=self.n_components,
                coords_sampling_dist="gm",
                mu_sampling_dist="io_ring",
                sigma_sampling_params=(0., 0.005),
                random_state=seed+1
            )
        elif self.group_id == "RC":
            self.coords_sampler = CoordsSampler(
                n_components=self.n_components,
                coords_sampling_dist="unf+gm",
                mu_sampling_dist="ring",
                sigma_sampling_params=(0.01, 0.01),
                random_state=seed+1
            )
        else:
            raise ValueError(f"unknown group_id: {self.group_id} (from {self.id}).")

        # demand sampler
        ### e.g.
        #'demand': {'dist': 'poisson', 'params': ([0.81], {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0})}
        p, p_misc = self.demand['params']   # tuple: (dist params, additional miscellaneous params)
        self.demand_p_misc = p_misc
        if self.demand['dist'] == "poisson":
            self.demand_sampler = stats.poisson(*p)
        elif self.demand['dist'] == "gamma":
            self.demand_sampler = stats.gamma(*p)
        else:
            raise ValueError(f"unknown demand_sampler cfg: {self.demand}.")

        # TW start sampler
        ### e.g.
        # 'tw_start': {'dist': 'KDE', 'params': <scipy.stats.kde.gaussian_kde object at 0x7ff5fe8c72e0>,
        # 'tw_start': {'dist': 'normal', 'params': (0.34984000000000004, 0.23766332152858588)}
        if self.tw_start['dist'] == "gamma":
            self.tw_start_sampler = stats.gamma(*self.tw_start['params'])
        elif self.tw_start['dist'] == "normal":
            self.tw_start_sampler = stats.norm(*self.tw_start['params'])
        elif self.tw_start['dist'] == "KDE":
            self.tw_start_sampler = self.tw_start['params']     # assigns fitted KDE model
        else:
            raise ValueError(f"unknown tw_start_sampler cfg: {self.tw_start}.")

        # TW len sampler
        if self.tw_len['dist'] == "const":
            self.tw_len_sampler = self.tw_len['params']     # this is the normalized len, there is also self.org_tw_len
        elif self.tw_len['dist'] == "gamma":
            self.tw_len_sampler = stats.gamma(*self.tw_len['params'])
        elif self.tw_len['dist'] == "normal":
            self.tw_len_sampler = stats.norm(*self.tw_len['params'])
        elif self.tw_len['dist'] == "KDE":
            self.tw_len_sampler = self.tw_len['params']     # assigns fitted KDE model
        else:
            raise ValueError(f"unknown tw_len_sampler cfg: {self.tw_len}.")

        # service time in Solomon data is constant for each instance, so mean == exact value
        self.service_time = self.norm_summary.loc['mean', 'service_time']

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rnd = np.random.default_rng(seed)
            self.coords_sampler.seed(seed+1)

    def sample(self, size: int, **kwargs) -> RPInstance:

        i = 0
        feasible = False
        while not feasible:
            if i > MAX_TRY:
                raise RuntimeError(f"Encountered many infeasible instances during sampling. "
                                   f"Try to adapt sampling parameters.")
            try:
                coords = self._sample_coords(size)
                # Euclidean distance
                dist_to_depot = dimacs_challenge_dist_fn_np(coords[1:], coords[0])
                time_to_depot = dist_to_depot / self.org_service_horizon

                demand = self._sample_demand(size)
                tw_start, tw_mask, num_tws = self._sample_tw_start(size, time_to_depot)
                tw = self._sample_tw_end(
                    size=size,
                    tw_start=tw_start,
                    time_to_depot=time_to_depot,
                    tw_mask=tw_mask,
                    num_tws=num_tws,
                )
            except AssertionError as ae:
                logger.debug(f"error while sampling. retrying... \n {ae}")
                i += 1
                continue
            feasible = True

        assert not np.any((tw[1:, 1] + time_to_depot + self.service_time) > 1.0)

        return RPInstance(
            coords=coords,
            demands=demand,
            tw=tw,
            service_time=self.service_time,
            graph_size=size+1,
            org_service_horizon=self.org_service_horizon,
            max_vehicle_number=25,
            vehicle_capacity=1.0,   # is normalized
            service_horizon=1.0,    # is normalized
            depot_idx=[0],
        )

    def _sample_coords(self, size: int):
        # simple case, all logic is already in coords sampler
        return self.coords_sampler.sample(n=size)

    def _sample_demand(self, size: int) -> np.ndarray:
        """sample demand according to cfg specifications."""
        # POISSON
        if self.demand['dist'] == "poisson":
            # sample from poisson dist
            smp = self.demand_sampler.rvs(size=size, random_state=self.rnd)
            mode = int(list(self.demand_p_misc.values())[0])
            keys = list(self.demand_p_misc.keys())
            smp = np.array([self.demand_p_misc[e] if e in keys else mode for e in smp])

        # GAMMA
        elif self.demand['dist'] == "gamma":

            # sample from gamma dist
            smp = self.demand_sampler.rvs(size=size, random_state=self.rnd)
            # round to bins
            unq, x_lim = self.demand_p_misc
            max_diff = np.max(unq[1:]-unq[:-1])
            if max_diff < 1 + np.finfo(float).eps:
                smp = np.round(smp)     # integer range bins
            else:
                bin_centers = np.array([(l+u)/2 for l, u in zip(unq[:-1], unq[1:])])
                smp = np.digitize(smp, bin_centers)
                smp = unq[smp]

            # truncate if sample value is larger than half a std
            # from the max observed in the training data (value saved in 'x_lim')
            # and set to median value
            m = self.demand_sampler.median()
            smp[smp > x_lim] = m
        else:
            raise RuntimeError

        # normalize demands
        smp /= self.vehicle_capacity

        # simple check of totals
        assert smp.sum() < 25, f"total sum of demands larger than total fleet capacity!"

        # add depot demand of 0
        return np.concatenate((np.zeros(1), smp), axis=0)

    def _sample_tw_start(self, size: int, time_to_depot: float
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
        """sample start time of TW according to cfg specifications."""

        # get fraction of TW
        if self.tw_frac < 1.0:
            num_tws = int(np.ceil(size*self.tw_frac))
            tw_mask = np.zeros(size, dtype=np.bool)
            tw_mask[self.rnd.choice(np.arange(size), size=num_tws, replace=False)] = 1
        else:
            num_tws = size
            tw_mask = np.ones(size, dtype=np.bool)

        # rejection sampling
        mean_tw_len = self.norm_summary.loc['mean', 'tw_len']
        eps = 1./self.org_service_horizon
        m = 10
        infeasible = True
        n = num_tws
        out = np.empty_like(time_to_depot)
        smp_idx = tw_mask.nonzero()[0]

        while infeasible:
            max_tw_start = 1. - np.repeat(time_to_depot[smp_idx] + self.service_time, m, axis=-1) - mean_tw_len/2
            assert np.all(max_tw_start > 0)

            if self.tw_start['dist'] == "gamma":
                smp = self.tw_start_sampler.rvs(size=m*n, random_state=self.rnd)
            elif self.tw_start['dist'] == "normal":
                smp = self.tw_start_sampler.rvs(size=m*n, random_state=self.rnd)
            elif self.tw_start['dist'] == "KDE":
                smp = self.tw_start_sampler.resample(size=m*n, seed=self.rnd)
            else:
                raise RuntimeError

            smp = smp.reshape(-1, m) + eps
            feasible = (smp > 0.0) & (smp <= max_tw_start.reshape(-1, m))
            has_feasible_val = np.any(feasible, axis=-1)
            # argmax returns idx of first True value if there is any, otherwise 0.
            first_feasible_idx = feasible[has_feasible_val].argmax(axis=-1)
            out[smp_idx[has_feasible_val]] = smp[has_feasible_val, first_feasible_idx]

            if np.all(has_feasible_val):
                infeasible = False
            else:
                no_feasible_val = ~has_feasible_val
                smp_idx = smp_idx[no_feasible_val]
                n = no_feasible_val.sum()
                m *= 2
            if m >= 320:   # 5
                # fall back to uniform sampling from valid interval
                s = eps
                e = max_tw_start
                out[smp_idx] = self.rnd.uniform(s, e, size=n)
                infeasible = False

        # set tw_start to 0 for nodes without TW
        out[~tw_mask] = 0

        return out, tw_mask, num_tws

    def _sample_tw_end(self,
                       size: int,
                       tw_start: np.ndarray,
                       time_to_depot: float,
                       tw_mask: np.ndarray,
                       num_tws: int,
                       ) -> np.ndarray:
        """sample end time of TW according to cfg specifications."""
        # make sure sampled end is feasible by checking if
        # service time + time to return to depot is smaller than total service horizon
        eps = 1./self.org_service_horizon
        t_delta = time_to_depot[tw_mask]
        inc_time = t_delta + self.service_time + eps
        smp_idx = tw_mask.nonzero()[0]
        out = np.empty_like(time_to_depot)

        if self.tw_len['dist'] == "const":
            assert np.all(inc_time + t_delta + self.tw_len_sampler < 1.0), \
                f"infeasible coordinates encountered"
            smp = self.tw_len_sampler   # all same constant value
            return_time = tw_start[tw_mask] + smp + inc_time
            infeasible = return_time >= 1.0
            if np.any(infeasible):
                inf_idx = smp_idx[infeasible]
                tw_start[inf_idx] = tw_start[inf_idx] - (return_time[infeasible] - 1 + eps)
                assert np.all(tw_start >= 0)

            out[tw_mask] = np.maximum(tw_start[tw_mask] + smp, t_delta + eps)

        else:
            # rejection sampling
            assert np.all(inc_time + t_delta < 1.0)
            m = 10
            infeasible = True
            n = num_tws

            while infeasible:
                if self.tw_len['dist'] == "gamma":
                    smp = self.tw_len_sampler.rvs(size=m*n, random_state=self.rnd)
                elif self.tw_len['dist'] == "normal":
                    smp = self.tw_len_sampler.rvs(size=m*n, random_state=self.rnd)
                elif self.tw_len['dist'] == "KDE":
                    smp = self.tw_len_sampler.resample(size=m*n, seed=self.rnd)
                else:
                    raise RuntimeError

                smp = smp.reshape(-1, m)
                # check feasibility
                # tw should be between tw_start + earliest possible arrival time from depot and
                # end of service horizon - time required to return to depot
                _tws = np.repeat(tw_start[smp_idx], m, axis=-1).reshape(-1, m)
                feasible = (
                    (_tws + np.repeat(t_delta, m, axis=-1).reshape(-1, m) < smp)
                    &
                    (_tws + np.repeat(inc_time, m, axis=-1).reshape(-1, m) + smp < 1.0)
                )
                has_feasible_val = np.any(feasible, axis=-1)
                # argmax returns idx of first True value if there is any, otherwise 0.
                first_feasible_idx = feasible[has_feasible_val].argmax(axis=-1)
                out[smp_idx[has_feasible_val]] = smp[has_feasible_val, first_feasible_idx]

                if np.all(has_feasible_val):
                    infeasible = False
                else:
                    no_feasible_val = ~has_feasible_val
                    smp_idx = smp_idx[no_feasible_val]
                    n = no_feasible_val.sum()
                    t_delta = t_delta[no_feasible_val]
                    inc_time = inc_time[no_feasible_val]
                    m *= 2
                if m >= 320:  # 5
                    # fall back to uniform sampling from valid interval
                    _tws = tw_start[smp_idx]
                    s = np.maximum(_tws, t_delta) + eps
                    e = 1. - inc_time

                    out[smp_idx] = self.rnd.uniform(s, e)
                    infeasible = False

        # add TW end as latest possible arrival time for all nodes without TW constraint
        out[~tw_mask] = 1.0 - time_to_depot[~tw_mask] - self.service_time - eps

        #assert np.all(out + time_to_depot + self.service_time < 1.0)
        return np.concatenate((
            np.array([[0, 1]]),     # add depot tw start 0 and end 1
            np.concatenate((tw_start[:, None], out[:, None]), axis=-1)
        ), axis=0)


class RPGenerator:
    """Generator unifying sampling procedure."""
    def __init__(self,
                 sample_cfg: Dict,
                 stats_file_path: str,
                 seed: int = 1,
                 ):
        self.sample_cfg = sample_cfg
        self.rnd = np.random.default_rng(seed)

        # load stats pkl
        with open(stats_file_path, 'rb') as f:
            stats_cfg = pickle.load(f)

        # get specified cfgs
        cfgs = []
        for grp in sample_cfg['groups']:
            for typ in sample_cfg['types']:
                for twf in sample_cfg['tw_fracs']:
                    cfgs.append(stats_cfg[grp][f"{grp}{typ}"][f"tw_frac={twf}"])
        self.cfgs = list(it.chain.from_iterable(cfgs))

        # initialize corresponding samplers
        self.num_samplers = len(self.cfgs)
        self.samplers = []
        for cfg in self.cfgs:
            self.samplers.append(InstanceSampler(cfg))

    def seed(self, seed: Optional[int] = None):
        """Set generator seed."""
        self.rnd = np.random.default_rng(seed)
        for i in range(self.num_samplers):
            self.samplers[i].seed(seed + i + 1)

    def generate(self, sample_size: int = 1000, graph_size: int = 100, **kwargs):
        """Generate data with corresponding RP generator function."""
        # sample from each instance sampler and take care of truncation if
        # sample_size % self.num_samplers != 0
        n_per_sampler = math.ceil(sample_size / self.num_samplers)
        num_trunc = self.num_samplers * n_per_sampler - sample_size
        assert self.num_samplers * n_per_sampler - num_trunc == sample_size
        smp = []
        for i in range(self.num_samplers):
            smp += self._sample(i, n_per_sampler-1 if i < num_trunc else n_per_sampler, graph_size)
        return smp

    def _sample(self, i: int, sample_size: int, graph_size: int = 100):
        sampler = self.samplers[i]
        return [sampler.sample(size=graph_size) for _ in range(sample_size)]

    @staticmethod
    def load_dataset(cfg: Dict,
                     filename: Optional[str] = None,
                     offset: int = 0,
                     limit: Optional[int] = None,
                     **kwargs):
        """Load data from file."""
        f_ext = os.path.splitext(filename)[1]
        assert f_ext in ['.pkl', '.dat', '.pt']
        filepath = os.path.normpath(os.path.expanduser(filename))
        logger.info(f"Loading dataset from:  {filepath}")

        try:
            data = torch.load(filepath, **kwargs)
        except RuntimeError:
            # fall back to pickle loading
            assert os.path.splitext(filepath)[1] == '.pkl', "Can only load pickled datasets."
            with open(filepath, 'rb') as f:
                data = pickle.load(f, **kwargs)

        # select instances specified in cfg
        instances = []
        for grp in cfg['groups']:
            for typ in cfg['types']:
                for twf in cfg['tw_fracs']:
                    instances.append(data[f"{grp}{typ}"][f"tw_frac={twf}"])
        data = list(it.chain.from_iterable(instances))

        if limit is not None and len(data) != (limit-offset):
            assert isinstance(data, List) or isinstance(data, np.ndarray), \
                f"To apply limit the data has to be of type <List> or <np.ndarray>."
            if len(data) < limit:
                warn(f"Provided data size limit={limit} but limit is larger than data size={len(data)}.")
            logger.info(f"Specified offset={offset} and limit={limit}. "
                        f"Loading reduced dataset of size={limit-offset}.")
            return data[offset:limit]
        else:
            return data

    @staticmethod
    def save_dataset(dataset: Union[List, np.ndarray],
                     filepath: str,
                     **kwargs):
        """Saves data set to file path"""
        filepath = format_ds_save_path(filepath, **kwargs)
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        logger.info(f"Saving dataset to:  {filepath}")
        try:
            torch.save(dataset, filepath)
        except RuntimeError:
            # fall back to pickle save
            assert os.path.splitext(filepath)[1] == '.pkl', "Can only save as pickle. Please add extension '.pkl'!"
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        return str(filepath)


class RPDataset(Dataset):
    """Routing problem dataset wrapper."""
    def __init__(self,
                 cfg: Dict,
                 data_pth: str = None,
                 stats_pth: str = None,
                 seed: int = None,
                 **kwargs):
        """

        Args:
            cfg: config of solomon stats to use for sampling
            data_pth: file path to load dataset
            stats_pth: file path to load  solomon stats
            seed: seed for random generator
            **kwargs:  additional kwargs for the generator
        """
        super(RPDataset, self).__init__()
        self.cfg = cfg
        self.data_pth = data_pth
        self.gen = None

        if data_pth is not None:
            logger.info(f"provided dataset {data_pth}, so no new samples are generated.")
        elif stats_pth is not None:
            self.gen = RPGenerator(sample_cfg=cfg, stats_file_path=stats_pth, seed=seed)
        else:
            RuntimeError(f"Need to specify either 'data_pth' or 'sample_cfg'.")

        self.size = None
        self.data = None

    def seed(self, seed: int):
        if self.gen is not None:
            self.gen.seed(seed)

    def load_ds(self, limit: Optional[int] = None, **kwargs):
        """Simply load dataset from data_path specified on init."""
        assert self.data_pth is not None
        self.data = RPGenerator.load_dataset(filename=self.data_pth,
                                             limit=limit,
                                             cfg=self.cfg,
                                             **kwargs)
        self.size = len(self.data)

    def sample(self, sample_size: int = 10000, graph_size: int = 100, **kwargs):
        """Loads fixed dataset if filename was provided else
        samples a new dataset based on specified config."""
        if self.data_pth is not None:   # load data
            self.data = RPGenerator.load_dataset(filename=self.data_pth,
                                                 limit=sample_size,
                                                 cfg=self.cfg,
                                                 **kwargs)
        else:
            self.data = self.gen.generate(
                sample_size=sample_size,
                graph_size=graph_size,
                **kwargs
            )
        self.size = len(self.data)
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============= #
# ### TEST #### #
# ============= #
def _test1():
    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"

    with open(LPATH, 'rb') as f:
        dset_cfg = pickle.load(f)

    cfgs = []
    for grp in SAMPLE_CFG['groups']:
        for typ in SAMPLE_CFG['types']:
            for twf in SAMPLE_CFG['tw_fracs']:
                cfgs.append(dset_cfg[grp][f"{grp}{typ}"][f"tw_frac={twf}"])
    cfgs = list(it.chain.from_iterable(cfgs))

    for cfg in cfgs:
        sampler = InstanceSampler(cfg)
        sampler.seed(123)
        for _ in range(100):
            sampler.sample(100)
    return True


def _test2():
    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    BS = 512
    N = 100

    gen = RPGenerator(sample_cfg=SAMPLE_CFG, stats_file_path=LPATH)
    gen.seed(123)
    data = gen.generate(BS, N)

    # for d in data:
    #     print(d.coords[0])

    assert len(data) == BS
    return True


def _test3():
    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    BS = 512
    N = 100

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(123)
    data = ds.sample(sample_size=BS, graph_size=N)
    assert len(data) == BS
    return True
