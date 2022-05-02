#
import sys
import logging
import time
from warnings import warn
from typing import Optional, Dict, Union, OrderedDict, TextIO, BinaryIO, List, Tuple, Any
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf as oc

import random
import numpy as np
import torch

from lib.routing import RPEnv, RPInstance
from lib.model import Policy
from lib.lns.gort import LocalSearch
from lib.lns.destructor import Destructor
from lib.utils import load_instance, to_rp_instance

logger = logging.getLogger("LNS")


#
class LNS:
    """
    Large Neighborhood Search based on an iterative scheme employing
    a pre-trained JAMPR+ policy model and optional local search via GORT.
    """

    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.debug_lvl = self.cfg.debug_lvl
        # set device
        if torch.cuda.is_available() and not cfg.cuda:
            warn(f"Cuda GPU is available but not used! Specify <cuda=True> in config file.")
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        self.type_twf_map_model = self.cfg['type_twf_map_model']
        self.num_steps = self.cfg.num_steps
        self.env = None
        self.policy = None
        self.ls = None
        self.ds = None
        self.org_instance = None
        self.rp_instance = None
        self.rnd = None
        self._ls_static = None

        self.best_solution = None
        self.best_solution_cost = None
        self.best_solution_step = None

        self.time_limit = None
        self.step = 0
        self.construction_time_max = 0

    def seed_all(self, seed: int):
        """Set seed for all pseudo random generators."""
        # will set some redundant seeds, but better safe than sorry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)

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

    def load_model(self, 
                   ckpt_pth: str, 
                   model_cfg: Optional[Union[DictConfig, Dict]] = None,
                   **kwargs):
        """load model from checkpoint."""
        # load checkpoint
        assert ckpt_pth is not None
        logger.info(f"loading model checkpoint: {ckpt_pth}")
        state_dict = torch.load(ckpt_pth, map_location=self.device)
        lns_env_cfg = self.cfg.env_cfg.copy()
        # override cfg
        self.cfg.update(state_dict["cfg"])

        # update cfg with additionally provided args
        if model_cfg is not None:
            model_cfg = oc.to_container(model_cfg, resolve=True) if isinstance(model_cfg, DictConfig) else model_cfg
            self.cfg.policy_cfg.update(model_cfg)

        # create env
        env_cfg = self.cfg.env_cfg.copy()
        num_samples = self.cfg.num_samples
        num_select = (self.cfg.num_best + self.cfg.num_other)
        assert num_samples > num_select, \
            f"num_samples must be larger than num_best + num_rnd " \
            f"but got {num_samples} <= {num_select}"
        assert num_samples % num_select == 0, f"num_samples must be divisible by num_best + num_rnd."
        if num_samples <= 1:
            warn("LNS should use 'num_samples' > 1")

        self.env = RPEnv(
            max_concurrent_vehicles=env_cfg.get('max_concurrent_vehicles'),
            k_nbh_frac=lns_env_cfg.get('k_nbh_frac'),
            pomo=self.cfg.pomo_inference,
            pomo_single_start_node=self.cfg.pomo_single_start_node,
            num_samples=num_samples,
            enable_render=lns_env_cfg.get('render'),
            plot_save_dir=lns_env_cfg.get('plot_save_dir'),
            device=self.device,
            debug=self.debug_lvl,
            inference=True,
            tour_graph_update_step=lns_env_cfg.get('tour_graph_update_step', 1)
        )

        # load model weights
        self._build_policy(state_dict)
        self.policy.eval()  # set to inference mode

    def load_instance(self, file: Union[str, TextIO, BinaryIO]):
        """Load problem instance."""
        self.org_instance = load_instance(file)
        self.rp_instance = to_rp_instance(self.org_instance)

    def init_ls(self):
        """Initialize GORT solver for local search."""
        self.ls = LocalSearch(**self.cfg.ls_cfg)
        # load static instance properties
        self.ls.load_instance(self.org_instance)

    def init_ds(self):
        """Initialize destructor."""
        self.ds = Destructor(
            num_partial=self.env.max_concurrent_vehicles,
            seed=self.cfg.seed,
            **self.cfg.ds_cfg
        )

    def setup(self, time_limit: Optional[int] = None, **kwargs):
        # make sure instance was loaded
        assert self.rp_instance is not None and isinstance(self.rp_instance, RPInstance), \
            f"Setup needs instance to be loaded first."
        # set time limit accordingly
        if time_limit is not None:
            self.time_limit = time_limit - 1  # 1s for setup, model + data loading etc.
        if (
                self.time_limit is not None
                and self.cfg.ls_step > 0
                and time_limit < 2 * self.cfg.ls_cfg.search_timelimit
        ):
            raise RuntimeError(f"specified general time_limit of {time_limit} is too small for a "
                               f"search_timelimit for local search of {self.cfg.ls_cfg.search_timelimit}")
        # check type and load
        # 1) corresponding JAMPR model for problem
        # 2) GORT with corresponding problem configuration
        typ, twf = self.rp_instance.type, self.rp_instance.tw_frac
        ckpt_pth = self.type_twf_map_model[int(typ)][float(twf)]
        self.load_model(ckpt_pth, **kwargs)
        self.init_ls()
        self.init_ds()

        ls_tm_per_iter = (
            self.cfg.ls_cfg.search_timelimit * (self.cfg.num_best + self.cfg.num_other)
        ) / self.cfg.ls_step
        logger.info(f"LS will take approx. {ls_tm_per_iter}s per iteration")
        # set seeds
        self.seed_all(self.cfg.seed)
        self.rnd = np.random.default_rng(self.cfg.seed + 1)

    def push_to_controller(self, solutions: List[List], costs: List[float], **kwargs):
        """Push the current best solution to the DIMACS controller."""
        sol, cost = self._select_best(solutions, costs)
        logger.info(f"best solution with cost: {self.best_solution_cost:.4f} (#{self.best_solution_step})")
        if not self.cfg.get('no_controller', False):
            # convert solution to TSPLIB solution format
            sol_str = self.format_tsplib(sol, cost)
            # push to stdout
            #logger.info(sol_str)
            try:
                sys.stdout.write(sol_str)
                sys.stdout.flush()
            except BrokenPipeError:
                warn(f"failed to push solution in iter: {self.step}.")
                sys.stderr.close()
                sys.exit()

    def _select_best(self, solutions: List[List], costs: List[float], **kwargs) -> Tuple[List, float]:
        """Select best solution."""
        assert len(solutions) == len(costs)
        if len(solutions) == 1:
            best = solutions[0]
            cost = costs[0]
        else:
            best_idx = np.argmin(costs)
            best = solutions[best_idx]
            cost = costs[best_idx]
        if self.best_solution is None or cost < self.best_solution_cost:
            self.best_solution = best
            self.best_solution_cost = cost
            self.best_solution_step = self.step

        return self.best_solution, self.best_solution_cost

    def _add_best(self,
                  solutions: List[List],
                  costs: List[float],
                  infos: Optional[List[Dict]] = None,
                  **kwargs) -> Tuple[List[List], List[float], Union[List[Dict], Any]]:
        """Always add best solution found so far to list of solutions by replacing worst."""
        if len(solutions) == 1:
            return [deepcopy(self.best_solution)], [self.best_solution_cost], [{}]
        if self.best_solution_cost in costs:
            worst_idx = np.argmax(costs)
            solutions[worst_idx] = deepcopy(self.best_solution)
            costs[worst_idx] = self.best_solution_cost
            if infos is not None:
                infos[worst_idx] = {}
        return solutions, costs, infos

    @staticmethod
    def format_tsplib(solution: List, cost: float) -> str:
        """Convert a solution and its corresponding cost to TSPLIB format."""
        sol_str = ""
        for i, tour in enumerate(solution):
            route = [str(t) for t in tour if t > 0]
            route = " ".join(route)
            sol_str += f"Route #{i}: {route}\n"
        sol_str += f"Cost {cost}\n"
        return sol_str

    @torch.no_grad()
    def _construct(self, partial_solutions: Optional[List[List]] = None,
                   **kwargs) -> Tuple[List[List], List[float]]:
        """Model inference to (re-)construct solutions."""
        logger.info(f"starting construction procedure...")
        t_start = time.time()
        if self.step == 0:
            # set to inference mode
            inf_mode = 'greedy' if self.cfg.pomo_inference else 'sampling'
            self.policy.set_decode_type(inf_mode)
            self.policy.reset_static()
            # load data into env
            self.env.load_data([self.rp_instance])
            # initial reset
            obs = self.env.reset()
        else:
            # load partial solution to reconstruct
            assert partial_solutions is not None
            obs = self.env.import_sol([partial_solutions])

        done = False
        #i = 0
        while not done:
            action, _, _ = self.policy(obs)
            # print(action[:, 0])
            obs, cost, done, info = self.env.step(action)
            #print(f"step: {i}")
            #i += 1

        solutions, costs = self.env.export_sol(self.cfg.num_best,
                                               self.cfg.num_other,
                                               mode=self.cfg.selection_mode)
        assert len(solutions) == 1
        # convert cost back to original scale
        cost = np.array(costs[0]) * self.rp_instance.org_service_horizon
        tm = time.time() - t_start
        if self.step > 0 and tm > self.construction_time_max:
            self.construction_time_max = tm
        logger.info(f"construction done after {tm: .3f}s.")
        return solutions[0], cost.tolist()

    def _local_search(self, solutions: List[List], costs: List[float],
                      **kwargs) -> Tuple[List[List], List[float], List[Dict]]:
        """Execute specified local search for all provided solutions."""
        logger.info(f"starting local search...")
        t_start = time.time()
        new_solutions, new_costs, infos = [], [], []
        # Added if-else for appending old costs to old solution if no improvement
        # is found otherwise would append None to new_costs.
        for sol, cost in zip(solutions, costs):
            plan, total_cost, info = self.ls.search(sol)
            new_solutions.append(plan)
            infos.append(info)
            if total_cost is None:
                new_costs.append(cost)
            else:
                new_costs.append(total_cost)
        logger.info(f"local search done after {time.time() - t_start: .3f}s.")
        return new_solutions, new_costs, infos

    def _destruct(self,
                  solutions: List[List],
                  costs: Optional[List[float]] = None,
                  infos: Optional[List[Dict]] = None,
                  **kwargs) -> List[List]:
        """Destruct provided solutions with specified operator."""
        logger.info(f"starting destruction procedure...")
        t_start = time.time()
        if self.cfg.pomo_inference:
            # bootstrap solutions to original num of POMO samples
            # and do a different destruction per sample
            n_expand = self.cfg.num_samples // len(solutions)
            solutions = solutions * n_expand
            costs = costs * n_expand if costs is not None else None
            infos = infos * n_expand if infos is not None else None
        solutions = self.ds.destruct(solutions, costs=costs, infos=infos, **kwargs)
        logger.info(f"destruction done after {time.time() - t_start: .3f}s.")
        return solutions

    def search(self) -> Tuple[List, float]:
        """Execute LNS."""
        infos = None
        t_start = None
        self.step = 0

        assert self.time_limit is not None or self.num_steps is not None
        if self.time_limit is not None:
            # use time limit instead of steps
            self.num_steps = float("inf")
            t_start = time.time()

        # construct
        solution, cost = self._construct()
        # in first iter directly push first solution to controller
        self.push_to_controller(solution, cost)

        while self.step < self.num_steps:
            # re-construct
            if self.step > 0:
                solution, cost = self._construct(solution)

            # do local search (every ls_step steps)
            if self.cfg.ls_step > 0 and self.step % self.cfg.ls_step == 0:
                solution, cost, infos = self._local_search(solution, cost)
            # push improved solution
            self.push_to_controller(solution, cost)

            # stop search if not enough time for another full iteration
            if self.time_limit is not None:
                if (
                    (time.time() - t_start) >=
                    max(self.time_limit -
                        (self.construction_time_max +
                            (self.cfg.num_best + self.cfg.num_other) *
                            self.cfg.ls_cfg.search_timelimit), 0)
                ):
                    break

            if self.cfg.add_best:
                solution, cost, infos = self._add_best(solution, cost, infos)
            # destruct solution
            solution = self._destruct(solution, cost, infos)

            self.step += 1

        # do last possibly longer LS on best solution found
        # using remaining time until limit
        if self.time_limit is not None:
            self.ls.search_timelimit = int(
                max(0.9*(self.time_limit - (time.time()-t_start)), 0)
            )
        else:
            self.ls.search_timelimit = 2 * self.ls.search_timelimit
        sol = [t[1:-1] for t in self.best_solution]
        solution, cost, infos = self._local_search([sol], [self.best_solution_cost])
        self.push_to_controller(solution, cost)
        return self.best_solution, self.best_solution_cost

    def search_iter(self, solution):
        """Execute one iteration of LNS."""
        infos = None
        # construct initially (when solution is None)
        if solution is None:  # and cost is None:
            # construct for first time
            solution, cost = self._construct()
            # set time_step
            self.step = 0
        else:
            # reconstruct
            solution, cost = self._construct(solution)

        # do local search (every ls_step steps)
        if self.cfg.ls_step > 0 and self.step % self.cfg.ls_step == 0:
            solution, cost, infos = self._local_search(solution, cost)

        sol_best, cost_best = self._select_best(solution, cost)
        logger.info(f"best solution with cost: {self.best_solution_cost:.4f} (#{self.best_solution_step})")

        # destruct solution
        solution_destroyed = self._destruct(solution, cost, infos)

        self.step += 1

        return sol_best, cost_best, solution_destroyed
