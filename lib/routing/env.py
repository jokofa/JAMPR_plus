#
import math
from typing import List, Tuple, Union, Optional
import time
import warnings
import logging
import numpy as np
import torch

from lib.routing.formats import RPInstance, RPObs
from lib.routing.visualization import Viewer
from lib.utils.graph_utils import GraphNeighborhoodSampler
from lib.utils.challenge_utils import dimacs_challenge_dist_fn
from lib.utils.seq_match import plan_to_string_seq, get_similarity_scores

logger = logging.getLogger("ENV")


class RPEnv:
    """
    RL simulation environment to solve CVRP-TW.
    Provides similar functionality to an OpenAI Gym environment,
    but is natively batched and can be run completely on GPU.
    """
    OBSERVATION_SPACE = {
        "node_features": 7,
        "tour_features": 5,
    }

    def __init__(self,
                 check_feasibility: bool = False,
                 max_concurrent_vehicles: int = 3,
                 k_nbh_frac: float = 0.25,
                 pomo: bool = False,
                 pomo_nbh_frac: float = 0.25,
                 pomo_single_start_node: bool = False,
                 num_samples: int = 1,
                 enable_render: bool = False,
                 plot_save_dir: Optional[str] = None,
                 device: Union[torch.device, str] = 'cpu',
                 fp_precision: torch.dtype = torch.float,
                 inference: bool = False,
                 tour_graph_update_step: int = 1,
                 debug: Union[bool, int] = False,
                 ):
        """

        Args:
            check_feasibility:  flag to check feasibility of updates
            max_concurrent_vehicles: max number of concurrently planned vehicles
            k_nbh_frac: fraction of graph_size defining size of node neighborhood
            pomo: flag to use POMO sampling
            pomo_nbh_frac: fraction of early TW to use as possible sampling candidates
            pomo_single_start_node: flag to fix only one single node per POMO rollout,
                                    independent of max_concurrent_vehicles
            num_samples: number of samples per instance
            enable_render: flag to enable rendering of environment
            plot_save_dir: directory to save rendered GIF files
            device: torch device to run env on
            fp_precision: floating point precision for float valued tensors
            tour_graph_update_step: number of steps when to update the tour edge graph
            inference: flag to put env in inference mode
            debug: flag to do additional checks and print additional debug information

        """
        self.check_feasibility = check_feasibility  # check feasibility of updates
        self.max_concurrent_vehicles = max_concurrent_vehicles
        self.k_nbh_frac = k_nbh_frac
        self.pomo = pomo
        if self.pomo:
            if num_samples <= 1:
                warnings.warn(f"POMO should use num_samples > 1")
            self._num_samples = num_samples
        elif inference:
            self._num_samples = num_samples
        else:
            self._num_samples = 1
        self.pomo_nbh_frac = pomo_nbh_frac
        self.pomo_single_start_node = pomo_single_start_node
        self.enable_render = enable_render
        self.plot_save_dir = plot_save_dir
        self.device = torch.device(device)
        self.fp_precision = fp_precision
        assert tour_graph_update_step != 0
        self.tour_graph_update_step = tour_graph_update_step
        self.inference = inference
        self.debug_lvl = 2 if isinstance(debug, bool) and debug else int(debug)
        if self.debug_lvl > 0:
            self.check_feasibility = True
        if self.debug_lvl > 2:
            warnings.simplefilter('always', RuntimeWarning)

        self.nbh_sampler = None
        self.bs = None
        self._bidx = None
        self._total = None

        self.coords = None
        self.demands = None
        self.tw = None
        self.service_time = None
        self.graph_size = None
        self.org_service_horizon = None
        self.max_vehicle_number = None
        self.vehicle_capacity = None
        self.service_horizon = None
        self.time_to_depot = None

        self._dist_mat = None

        self._visited = None  # nodes that have been visited (served)
        self._finished = None   # vehicles which have started a tour and returned to the depot
        self.tour_plan = None
        self.active_vehicles = None  # which vehicles are currently active
        self.active_to_plan_idx = None    # map absolut vehicle idx to idx in vehicle buffer
        self.next_index_in_tour = None  # index at which to add next node to plan for each active vehicle

        # dynamic vehicle features
        self.cur_node = None
        self.cur_cap = None
        self.cur_time = None
        self.cur_time_to_depot = None

        # graph buffers
        self.k_nbh_size = None
        self.depot_idx = None
        self._tour_batch_idx = None
        self.nbh_edges, self.nbh_weights = None, None
        self.tour_edges, self.tour_weights = None, None
        self.ordered_idx = None

        self.viewer = None
        self.render_buffer = None

        self._zero = torch.zeros(1, dtype=torch.long, device=self.device)
        self._one = torch.ones(1, dtype=self.fp_precision, device=self.device)
        self._has_instance = False
        self._is_reset = False
        self._step = None
        self._render_cnt = 0

    def seed(self, seed: int) -> List[int]:
        torch.manual_seed(seed)
        return [seed]

    def reset(self) -> RPObs:
        """Reset the simulator and return the initial state."""
        assert self._has_instance, f"need to load instance first."
        self._step = 0

        # reset graph buffers
        self.depot_idx = None
        self.ordered_idx = None
        self.nbh_edges, self.nbh_weights = None, None
        self.tour_edges, self.tour_weights = None, None
        self._tour_batch_idx = None
        # create graph attributes
        self.to_graph()

        # reset other buffers
        self._visited = torch.zeros(self.bs, self.graph_size,
                                    dtype=torch.bool, device=self.device)
        self._finished = torch.zeros(self.bs, self.max_vehicle_number,
                                     dtype=torch.bool, device=self.device)
        seq_buffer_len = min(64, self.graph_size)
        self.tour_plan = torch.zeros(self.bs, self.max_vehicle_number, seq_buffer_len,
                                     dtype=torch.int16, device=self.device)
        self.active_vehicles = torch.zeros(self.bs, self.max_vehicle_number,
                                              dtype=torch.bool, device=self.device)
        # set first max_concurrent vehicles as active
        self.active_vehicles[:, :self.max_concurrent_vehicles] = 1
        self.active_to_plan_idx = self.active_vehicles.nonzero(as_tuple=True)[1].view(self.bs, -1)
        self.next_index_in_tour = torch.zeros(self.bs, self.max_concurrent_vehicles,
                                              dtype=torch.long, device=self.device)

        self.cur_node = torch.zeros(self.bs, self.max_concurrent_vehicles,
                                    dtype=torch.long, device=self.device)
        self.cur_cap = torch.full((self.bs, self.max_concurrent_vehicles), self.vehicle_capacity,
                                  dtype=self.fp_precision, device=self.device)
        self.cur_time = torch.zeros(self.bs, self.max_concurrent_vehicles,
                                    dtype=self.fp_precision, device=self.device)
        self.cur_time_to_depot = torch.zeros(self.bs, self.max_concurrent_vehicles,
                                             dtype=self.fp_precision, device=self.device)

        if self.enable_render:
            if self.viewer is not None:
                self.viewer.save()
                self.viewer.close()
                self.viewer = None
            self.render_buffer = {}
            self._render_cnt += 1

        self._total = torch.zeros(self.bs, dtype=self.fp_precision, device=self.device)

        # POMO start node sampling
        if self.pomo:
            self._reset_pomo()

        self._is_reset = True

        return self._get_observation()

    def step(self, action: torch.Tensor):
        """Take an action and do one step in the environment.

        Args:
            action: (BS, 2) - selected tour idx, selected node idx

        Returns:
            - observations,
            - reward (cost),
            - done,
            - info dict
        """
        assert self._is_reset
        assert action.size(0) == self.bs and action.size(1) == 2

        # action is selected tour and next node for that tour
        tour_select = action[:, 0]
        next_node = action[:, 1]
        ret_mask = (next_node == self.depot_node)
        cost = self._update(tour_select, next_node, ~ret_mask)

        # convert selection idx to internal tour plan idx over max vehicles
        tour_plan_select = self.active_to_plan_idx[self._bidx, tour_select]
        # add next node in tour
        nxt = self.next_index_in_tour[self._bidx, tour_select]
        try:
            self.tour_plan[self._bidx, tour_plan_select, nxt] = next_node.to(torch.int16)
        except IndexError:
            # this can also happen when the depot is not properly masked during inference
            inf_msk = (nxt >= 64)
            n_inf = inf_msk.sum()
            inf_tours = self.tour_plan[inf_msk][self.active_vehicles[inf_msk]]
            raise RuntimeError(f"Current rollout could not solve at least {n_inf} instances."
                               f"\nmax_len tours: {inf_tours}")

        # increase idx of next position in tour
        # (mask if selects depot, otherwise might exceed max seq len!)
        not_ret = ~ret_mask
        self.next_index_in_tour[self._bidx[not_ret], tour_select[not_ret]] = nxt[not_ret] + 1

        all_visited = self.visited.all(-1)
        self._visited[self._bidx, next_node] = 1
        # depot node is never marked as visited!
        self._visited[:, 0] = 0

        # manage returning tours (and init new tour on return)
        ret_mask = ret_mask & ~all_visited
        if ret_mask.any():
            nxt_active = self._get_next_active_vehicle()[ret_mask]
            all_started = (nxt_active == 0)
            ret_idx = ret_mask.nonzero(as_tuple=True)[0]
            if all_started.any():
                warnings.warn("used all available vehicles!", RuntimeWarning)
                # remove idx of instance without further vehicles from indices to reset
                ret_idx = ret_idx[~all_started]
                nxt_active = nxt_active[~all_started]

            ret_tour_idx = tour_select[ret_idx]
            # reset buffers
            self.cur_node[ret_idx, ret_tour_idx] = self.depot_node[ret_idx]
            self.cur_cap[ret_idx, ret_tour_idx] = self.vehicle_capacity
            if self.check_feasibility:
                assert (self.cur_time[ret_idx, ret_tour_idx] <= 1.0).all()
            self.cur_time[ret_idx, ret_tour_idx] = 0
            assert (self.cur_time_to_depot[ret_idx, ret_tour_idx] == 0).all()
            # update active and returned vehicles
            cur_active = self.active_to_plan_idx[ret_idx, ret_tour_idx]
            self._finished[ret_idx, cur_active] = 1
            self.active_vehicles[ret_idx, cur_active] = 0
            self.active_vehicles[ret_idx, nxt_active] = 1
            self.next_index_in_tour[ret_idx, ret_tour_idx] = 0
            # set active-plan idx map of currently returning vehicle to
            # the plan idx of the next vehicle
            self.active_to_plan_idx[ret_idx, ret_tour_idx] = nxt_active

        # stop if
        # 1) all nodes were visited or
        # 2) all vehicles were used, in which case there might still be unvisited nodes!
        done = self.visited.all() or self._step >= self.graph_size + self.max_vehicle_number + 1
        if done:
            # make sure all vehicles return to depot
            added_cost = self._return_all()
            cost += added_cost
            # add cost for singleton tours to remaining unvisited nodes
            if not self.visited.all():
                s_bs_idx, s_nd_idx = (~self.visited).nonzero(as_tuple=True)
                singleton_cost = self.time_to_depot[:, 1:][s_bs_idx, s_nd_idx] * 2
                bs_idx, cnt_per_idx = s_bs_idx.unique(return_counts=True)
                n_unq = len(bs_idx)
                cst = torch.empty((n_unq, ), device=self.device, dtype=singleton_cost.dtype)
                for i, (s, e) in enumerate(zip(self._cumsum0(cnt_per_idx), cnt_per_idx.cumsum(dim=-1))):
                    cst[i] = singleton_cost[s:e].sum()
                cost[bs_idx] = cost[bs_idx] + cst

            self._has_instance = False
            self._is_reset = False

        # update graph data
        self.to_graph()
        self._total += cost

        info = {
            'current_total_cost': self._total.cpu().numpy(),
            'k_used': self.k_used.cpu().numpy() if done and len(self.k_used) > 0 else [-1],
            'max_tour_len': self.tour_plan.argmin(dim=-1).max().item(),
        }
        self._step += 1

        return self._get_observation() if not done else None, cost, done, info

    def render(self, as_gif: bool = True, **kwargs):
        assert self.enable_render, f"Need to specify <enable_render=True> on init."
        if as_gif:
            assert self.plot_save_dir is not None, f"need to specify directory to save gifs."
            if self._step >= 155:
                return  # can only render max ~155 steps as GIF

        b_idx = 0
        if self.viewer is None:
            if self.bs != 1 and self.debug_lvl > 1:
                warnings.warn(f"batch_size > 1. Will only render batch instance with idx={b_idx}.")
            # create new viewer object
            self.viewer = Viewer(
                locs=self.coords[b_idx].cpu().numpy(),
                save_dir=self.plot_save_dir,
                gif_naming=f"render_ep{self._render_cnt}",
                as_gif=as_gif,
                add_idx=False,
                **kwargs
            )

        # update buffer and render new tour
        self.render_buffer['edges'] = self._get_edges_to_render(b_idx)
        self.viewer.update(
            buffer=self.render_buffer,
            cost=self._total[b_idx].cpu().item(),
            n_iters=self._step,
            **kwargs
        )
        return self.viewer.render_rgb()

    def load_data(self, batch: List[RPInstance]) -> None:
        """Load a list of RPInstances into the environment."""
        self.bs = len(batch)
        self._bidx = torch.arange(self.bs, device=self.device)

        self.coords = self._stack_to_tensor(batch, 'coords')
        self.demands = self._stack_to_tensor(batch, 'demands')
        self.tw = self._stack_to_tensor(batch, 'tw')
        self.service_time = self._stack_to_tensor(batch, 'service_time')

        gs = batch[0].graph_size
        assert np.all(np.array([x.graph_size for x in batch]) == gs)
        self.graph_size = gs
        self.org_service_horizon = self._stack_to_tensor(batch, 'org_service_horizon')

        k = batch[0].max_vehicle_number
        assert np.all(np.array([x.max_vehicle_number for x in batch]) == k)
        # provide slightly more vehicles to make sure we always get a solution for all nodes
        self.max_vehicle_number = int(k + np.floor(np.log(k)))  #int(k + np.floor(np.sqrt(gs)))
        assert np.all(np.array([x.vehicle_capacity for x in batch]) == 1)
        self.vehicle_capacity = batch[0].vehicle_capacity   # normed to 1.0
        assert np.all(np.array([x.service_horizon for x in batch]) == 1)
        self.service_horizon = batch[0].service_horizon     # normed to 1.0
        assert np.all(np.array([x.depot_idx[0] for x in batch]) == 0)

        if self.inference:
            # compute and keep full distance matrix in memory
            self._dist_mat = self.compute_distance_matrix(self.coords)
            t_delta = self._dist_mat[:, :, 0]
        else:
            idx_pair = torch.stack((
                torch.arange(0, self.graph_size, device=self.device)[None, :].expand(self.bs, self.graph_size),
                self.depot_node[:, None].expand(self.bs, self.graph_size)
            ), dim=-1).view(self.bs, -1)
            idx_coords = self.coords.gather(
                dim=1, index=idx_pair[:, :, None].expand(self.bs, -1, 2)
            ).view(self.bs, -1, 2, 2)
            t_delta = (
                    dimacs_challenge_dist_fn(idx_coords[:, :, 0, :], idx_coords[:, :, 1, :]) /
                    self.org_service_horizon[self._bidx][:, None]
            )

        self.time_to_depot = t_delta
        if self.debug_lvl > 0:
            assert (t_delta >= 0).all()
            if ((self.tw[:, :, 1] + self.time_to_depot + self.service_time[:, None])[:, 1:] > 1.0).any():
                msg = f"cannot return to the depot when arriving within TW of some customers."
                if self.inference:
                    warnings.warn(msg + f" Applying fix during inference...")
                else:
                    raise RuntimeError(msg)
        if self.inference:
            # quick and dirty fix for instances where it is not guaranteed that
            # one can return to the depot when arriving within any TW of a customer
            return_time = (self.tw[:, :, 1] + self.time_to_depot + self.service_time[:, None])
            no_return_mask = (return_time > 1.0)
            no_return_mask[:, 0] = False    # always false für depot
            if (no_return_mask.sum(-1) > self.graph_size * 0.05).any():
                warnings.warn(f"Need to fix many TW for return to depot. Consider checking instance.")
            delta = return_time[no_return_mask] - 1.0
            new_tw = torch.stack((
                self.tw[no_return_mask][:, 0],
                self.tw[no_return_mask][:, 1]-delta
            ), axis=-1)
            assert (new_tw[:, 1] - new_tw[:, 0] > 0.005).all()
            self.tw[no_return_mask] = new_tw

        if self.num_samples > 1 or self.pomo:
            self._init_sampling()

        # init nbh graph sampler if not existing yet (assumes all instances have same graph_size)
        if self.nbh_sampler is None:
            self.nbh_sampler = GraphNeighborhoodSampler(self.graph_size, k_frac=self.k_nbh_frac)

        self._has_instance = True

    def clear_cache(self):
        """Clear all object references to tensors."""
        self.bs = None
        self._bidx = None
        self._total = None

        self.coords = None
        self.demands = None
        self.tw = None
        self.service_time = None
        self.graph_size = None
        self.org_service_horizon = None
        self.max_vehicle_number = None
        self.vehicle_capacity = None
        self.service_horizon = None
        self.time_to_depot = None

        self._visited = None
        self._finished = None
        self.tour_plan = None
        self.active_vehicles = None
        self.active_to_plan_idx = None
        self.next_index_in_tour = None

        self.cur_node = None
        self.cur_cap = None
        self.cur_time = None
        self.cur_time_to_depot = None

        self.k_nbh_size = None
        self.depot_idx = None
        self._tour_batch_idx = None
        self.nbh_edges, self.nbh_weights = None, None
        self.tour_edges, self.tour_weights = None, None
        self.ordered_idx = None

        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.render_buffer = None

        self._has_instance = False
        self._is_reset = False
        self._step = None

    def export_sol(self, num_best: int = 3, num_other: int = 3, mode: str = "random") -> Tuple[List[List], List]:
        """
        Export the current tour-plans as list (of lists of lists).
        -> (original BS, num_best + num_other, max_num_vehicles, max_seq_len)
        """

        if self._num_samples > 1:
            # select "promising" tours
            # first select num_best best solutions
            n_smp = num_best + num_other
            assert self._num_samples > n_smp, \
                f"specified selection of total of {n_smp} samples but env was configured " \
                f"with num_samples = {self._num_samples} < {n_smp}!"
            cost_idx = self.total_cost.view(-1, self._num_samples).sort(dim=-1).indices
            _bs = cost_idx.size(0)
            idx = cost_idx[:, :num_best]
            t_start = time.time()

            if mode == "random":
                logger.info(f"export with random sampling...")
                # num_other random solutions
                if num_other > 0:
                    rnd_idx = self._randint(bs=_bs, n=num_other, high=self._num_samples-num_best)
                    rnd_idx = cost_idx[:, num_best:].gather(dim=-1, index=rnd_idx)
                    idx = torch.cat((idx, rnd_idx), dim=-1)
                # reshape over sample dimension and select samples at index
                # from (org BS * n_samples, max_num_vehicles, max_seq_len)  ->
                tp = self.tour_plan.view(-1, self._num_samples, self.max_vehicle_number, self.tour_plan.size(-1)).gather(
                    dim=1, index=idx[:, :, None, None].expand(_bs, n_smp, self.max_vehicle_number, self.tour_plan.size(-1))
                )
                tours = self._sol_to_list(tp)
                costs = self.total_cost.view(-1, self._num_samples).gather(dim=-1, index=idx).cpu().tolist()
                
            elif mode == "similarity":
                logger.info(f"export with similarity sampling...")
                # num_other solutions which are most dissimilar to the best solutions selected
                all_tours = self._sol_to_list(
                    self.tour_plan.view(-1, self._num_samples, self.max_vehicle_number, self.tour_plan.size(-1))
                )
                tours = []
                costs = []
                for bst, smps, cst in zip(
                        idx,
                        all_tours,
                        self.total_cost.view(-1, self._num_samples).cpu().tolist()
                ):
                    # select best
                    best_smps = [smps[i] for i in bst]
                    best_costs = [cst[i] for i in bst]
                    # select others
                    other_smps = [smps[i] for i in range(len(smps)) if i not in bst]
                    other_costs = [cst[i] for i in range(len(smps)) if i not in bst]
                    # calculate similarity scores of plans
                    # via (sub-)sequence matching
                    b = [plan_to_string_seq(p) for p in other_smps]
                    div_smp_idx = []
                    for smp in best_smps:
                        scores = get_similarity_scores(
                            anchor=plan_to_string_seq(smp),
                            candidates=b,
                        )
                        div_smp_idx.append(np.argsort(scores))  # get indices of lowest scores

                    # cyclic selection of most diverse other sample for each of the best samples
                    idx = []
                    i = 0
                    j = [0]*num_best
                    while len(idx) < num_other:
                        div_idx = div_smp_idx[i][j[i]]
                        if div_idx not in idx:
                            idx.append(div_idx)
                        else:
                            j[i] += 1
                            continue
                        if i < num_best-1:
                            i += 1
                        else:
                            i = 0

                    tours.append(best_smps + [other_smps[i] for i in idx])
                    costs.append(best_costs + [other_costs[i] for i in idx])
            else:
                raise ValueError(f"unknown selection mode: {mode}.")
            logger.info(f"export done after {time.time() - t_start: .3f}s.")
            return tours, costs
        else:
            tp = self.tour_plan.view(-1, self._num_samples, self.max_vehicle_number, self.tour_plan.size(-1))
            return self._sol_to_list(tp), self.total_cost.clone().cpu().tolist()

    def _sol_to_list(self, tp: torch.Tensor):
        # handles solutions which contain unvisited nodes as singleton tours
        # for loop is OK here, normally we only export solutions for small amounts of instances
        tours = []
        full_set = set(range(self.graph_size))
        for plans in tp:
            tour_set = []
            for plan in plans:
                unq = torch.unique(plan)
                singletons = []
                # if len(unq) != self.graph_size:
                #     singletons = [[0, e, 0] for e in full_set - set(unq.cpu().tolist())]
                # plan = plan[plan.sum(-1) > 0].cpu()
                # tour_set.append([[0] + tp[tp > 0].tolist() + [0] for tp in plan] + singletons)
                if len(unq) != self.graph_size:
                    singletons = [[e] for e in full_set - set(unq.cpu().tolist())]
                plan = plan[plan.sum(-1) > 0].cpu()
                tour_set.append([tp[tp > 0].tolist() for tp in plan] + singletons)
            tours.append(tour_set)
        return tours

    def import_sol(self, sol: List[List], cost: Optional[List] = None) -> RPObs:
        """Import partial RP solutions represented in a list of lists format
        and update buffers and state correspondingly."""
        # CONVENTION: all 'complete' tours start and end at the depot idx.
        # all other tours are considered partial tours or singletons
        # resulting from the destruction procedure.
        # sol: (BS, num_samples, num_tours, ...)

        assert self.inference, f"Can only import solutions in inference mode."
        # check dims
        org_bs = self.bs // self.num_samples    # number of instances
        assert len(sol) == org_bs
        n_samples = np.array([len(s) for s in sol])
        assert np.all(n_samples == n_samples[0])
        n_samples = n_samples[0]
        if self.pomo:
            assert self.num_samples == n_samples, f"imported solutions for POMO must include num_samples " \
                                                  f"samples, but got {n_samples} != {self.num_samples}"
        else:
            assert self.num_samples % n_samples == 0
        sample_fact = self.num_samples // n_samples

        # reset buffers
        # instead of re-creating the tensors we just fill them with the correct reset values
        self._visited.fill_(value=0)
        self._finished.fill_(value=0)
        self.tour_plan.fill_(value=0)
        self.active_vehicles.fill_(value=0)
        self.next_index_in_tour.fill_(value=0)

        self.cur_node.fill_(value=0)
        self.cur_cap.fill_(value=self.vehicle_capacity)
        self.cur_time.fill_(value=0)
        self.cur_time_to_depot.fill_(value=0)

        recompute_cost = False
        if cost is not None:
            self._total = torch.tensor(cost, dtype=self._total.dtype, device=self.device)
        else:
            recompute_cost = True
            total_costs = []

        # read in solutions
        bs_idx = 0
        for inst_sol in sol:
            for smp in inst_sol:
                # partial tours are all non-singleton tours which do not start and end at the depot
                num_partial = 0
                t_idx = 0
                service_tm = self.service_time[bs_idx]
                costs = []
                for tour in smp:
                    l = len(tour)
                    if l > 1:
                        try:
                            # complete
                            if tour[0] == tour[-1] == self.depot_node[0]:   # starts and ends at depot node
                                # just add to tour plan
                                self.tour_plan[bs_idx, t_idx, :l-1] = torch.tensor(
                                    tour[1:], dtype=self.tour_plan.dtype, device=self.device
                                )
                                self._finished[bs_idx, t_idx] = True
                                if recompute_cost:
                                    costs.append(self._recompute_cost(tour, bs_idx, service_tm))

                            # partial
                            else:
                                t = torch.tensor(tour, dtype=torch.long, device=self.device)
                                self.tour_plan[bs_idx, t_idx, :l] = t.to(dtype=self.tour_plan.dtype)
                                # add to cumulative buffers
                                self.cur_node[bs_idx, num_partial] = t[-1]
                                self.cur_cap[bs_idx, num_partial] = (
                                    1.0 - self.demands[bs_idx].gather(dim=-1, index=t).sum()
                                )
                                self.cur_time_to_depot[bs_idx, num_partial] = self.time_to_depot[bs_idx, t[-1]]

                                # recalculate current time of vehicle
                                tm = self._recompute_cost(t, bs_idx, service_tm)
                                if recompute_cost:
                                    costs.append(tm)
                                self.cur_time[bs_idx, num_partial] = tm
                                self.next_index_in_tour[bs_idx, num_partial] = len(t)
                                self.active_vehicles[bs_idx, t_idx] = True

                                num_partial += 1

                        except IndexError:
                            raise RuntimeError(f"Number of tours of provided solution "
                                               f"is larger than max_num_vehicles!")

                        t_idx += 1

                    # singleton tour
                    else:
                        pass    # nothing to do in this case

                # check if number of partial tours <= max_concurrent_vehicles
                assert num_partial <= self.max_concurrent_vehicles
                must_assign = self.max_concurrent_vehicles-num_partial
                if must_assign > 0:
                    # start a new tour for each non existing partial tour
                    for i in range(num_partial, num_partial+must_assign):
                        nxt_active = self._get_next_active_vehicle()[bs_idx]
                        self.active_vehicles[bs_idx, nxt_active] = 1
                        self.active_to_plan_idx[bs_idx, i] = nxt_active

                # adapt visitation status
                nz = self.tour_plan[bs_idx].nonzero(as_tuple=True)
                self._visited[bs_idx, self.tour_plan[bs_idx, nz[0], nz[1]].long()] = 1
                if recompute_cost:
                    total_costs.append(sum(costs))
                # inc per sample
                bs_idx += sample_fact

        if recompute_cost:
            self._total = torch.tensor(total_costs, dtype=self.fp_precision, device=self.device)

        # re-expand if the number of samples changed during selection procedure
        # POMO sampling will always do the expansion during the destruction procedure,
        # but standard sampling needs it here explicitly
        if n_samples != self.num_samples:
            self._expand_sample_dimension(sample_fact)
        #
        self.active_to_plan_idx = self.active_vehicles.nonzero(as_tuple=True)[1].view(self.bs, -1)
        # re-create graph
        self.to_graph()

        self._has_instance = True
        self._is_reset = True
        self._step = (self._visited.sum(-1) + self._finished.sum(-1)).max().cpu().item()

        return self._get_observation()

    def _recompute_cost(self, tour: Union[List, torch.Tensor], bs_idx: int, service_time: float):
        # recalculate current time of vehicle
        tm = 0
        prev = 0
        for nxt in tour:
            # select from distance matrix
            tm += self._dist_mat[bs_idx, prev, nxt]
            # add waiting time and service time
            tm += ((self.tw[bs_idx, nxt][0] - tm).clamp_(min=0) + service_time)
            prev = nxt
        return tm.cpu().item()

    def destruct(self, **kwargs):
        """Tensor-based native destruction operator circumventing
        expensive conversion to lists during solution export/import."""
        raise NotImplementedError

    @staticmethod
    def _cumsum0(t: torch.Tensor) -> torch.Tensor:
        """calculate cumsum of t starting at 0."""
        return torch.cat((
            torch.zeros(1, dtype=t.dtype, device=t.device),
            torch.cumsum(t, dim=-1)[:-1]
        ), dim=0)

    @property
    def depot_node(self) -> torch.Tensor:
        """idx of depot node is always 0."""
        if self._zero.device != self.device:
            self._zero = self._zero.to(device=self.device)
        return self._zero[:, None].expand(-1, self.bs).view(-1)

    @property
    def idx_inc(self) -> torch.Tensor:
        """Returns the index increase necessary to
        transform to BS x N running index."""
        assert self.depot_idx is not None and len(self.depot_idx) == self.bs
        return self.depot_idx

    @property
    def visited(self) -> torch.BoolTensor:
        """Returns mask for all nodes without depot (BS, N-1),
        indicating if the respective node was already visited."""
        return self._visited[:, 1:]

    @property
    def k_used(self) -> torch.Tensor:
        """Returns the number of vehicles used for each instance."""
        _active = self.active_vehicles.clone()
        _active[self.active_vehicles] = (self.cur_node != self.depot_node[:, None]).view(-1)
        return (self._finished | _active).sum(-1)

    @property
    def total_cost(self) -> torch.Tensor:
        """return the current total cost of the solution."""
        return self._total.clone()

    @property
    def num_samples(self):
        return self._num_samples

    def to_graph(self) -> None:
        """Create static nbh graph and dynamic tour graph components."""
        if self.depot_idx is None:
            # starting node indices of each batch instance are exactly depot
            self.depot_idx = self._cumsum0(
                torch.from_numpy(np.full(self.bs, self.graph_size))
                    .to(dtype=torch.long, device=self.device)
            )

        if self._tour_batch_idx is None:
            self._tour_batch_idx = torch.arange(self.bs, device=self.device)[:, None].expand(-1, self.max_vehicle_number)

        # nbh graph is static and only needs to be created at start of episode
        if self.nbh_edges is None or self.nbh_weights is None:
            nbh_edges, nbh_weights = [], []
            for i, c in enumerate(self.coords):
                e = self.nbh_sampler(c)
                nbh_edges.append(e + self.idx_inc[i])   # increase node indices by running idx
                # calculate weights
                idx_coords = c[e]
                nbh_weights.append(
                    dimacs_challenge_dist_fn(idx_coords[0], idx_coords[1])/self.org_service_horizon[i]
                )
            self.nbh_edges = torch.cat(nbh_edges, dim=-1)
            self.nbh_weights = torch.cat(nbh_weights, dim=-1)
            self.k_nbh_size = self.nbh_sampler.k

        if self.tour_edges is None or self.tour_weights is None:
            # initialize - no tours exist
            # create just dummy edges from depot to depot
            self.tour_edges = torch.cat((self.depot_idx[None, :], self.depot_idx[None, :]), dim=0)
            self.tour_weights = torch.zeros(self.bs, dtype=self.fp_precision, device=self.device)
        elif (self._step <= self.max_concurrent_vehicles+1) or (self._step % self.tour_graph_update_step == 0):
            # infer edges from current routes
            # select all routes which are either finished or active (partial solutions)
            selection_mask = self._finished | self.active_vehicles
            # increase to running idx and get corresponding node indices
            #tours = (self.tour_plan + self.idx_inc[:, None, None])[selection_mask]
            tours = (
                self.tour_plan[selection_mask] +
                self.idx_inc[:, None, None].expand(-1, selection_mask.size(-1), 1)[selection_mask]
            )
            if self.debug_lvl > 1:
                assert (tours[:, -1] == self.depot_idx.repeat_interleave(selection_mask.sum(-1), dim=-1)).all()
            sbl = tours.size(-1)
            tours = tours.view(-1, sbl)  # (BS, max_concurrent, seq_buffer_len) -> (-1, seq_buffer_len)
            # create edges as node idx pairs
            # automatically adds an edge from the last node back to the depot
            tours = torch.cat((
                torch.roll(tours, shifts=1, dims=-1)[:, None, :],   # cyclic shift by 1
                tours[:, None, :]
            ), axis=1).permute(1, 0, 2).reshape(2, -1)
            tour_batch_idx = self._tour_batch_idx[selection_mask]
            # remove dummies (depot self loops)
            selection_mask = (tours[0, :] != tours[1, :])
            self.tour_edges = tours[:, selection_mask]
            # get weights
            # TODO: better way than with tour_batch_idx which is only used here?!
            tour_batch_idx = (
                tour_batch_idx[:, None].expand(-1, sbl).reshape(-1)
            )[selection_mask]
            if self.inference:
                # select from distance matrix
                idx = self.tour_edges - self.idx_inc[tour_batch_idx]
                self.tour_weights = self._dist_mat[tour_batch_idx][
                    torch.arange(tour_batch_idx.size(0), device=self.device), idx[0,], idx[1,]
                ]
            else:
                # compute on the fly
                idx_coords = self.coords.view(-1, 2)[self.tour_edges]
                self.tour_weights = (
                        dimacs_challenge_dist_fn(idx_coords[0], idx_coords[1]) /
                        self.org_service_horizon[tour_batch_idx]
                )
        else:
            # no update to tour graph
            self.tour_edges = torch.empty(0)
            self.tour_weights = torch.empty(0)

    def get_node_nbh(self, node_idx: torch.Tensor) -> torch.LongTensor:
        """Return the neighborhood of the specified nodes."""
        assert node_idx.size(0) == self.bs
        depot_mask = (node_idx == 0)
        if depot_mask.any():
            # first N elements in self.nbh_edges[0] are depot nbh
            depot_nbh = self.nbh_edges.view(2, self.bs, -1)[:, :, :self.graph_size]
            if self.ordered_idx is None:
                # order the nodes in the depot nbh by their distance to depot
                idx_coords = self.coords.view(-1, 2)[depot_nbh.reshape(2, -1)]
                # here euclidean distance is sufficient
                self.ordered_idx = torch.norm(idx_coords[0]-idx_coords[1], p=2, dim=-1)\
                    .view(self.bs, -1)\
                    .argsort(dim=-1, descending=False)

            # first check visitation status
            vis_mask = ~self._visited.gather(dim=-1, index=self.ordered_idx)
            # get mask of the first 'nbh_size' closest unvisited nodes
            _msk = vis_mask.cumsum(dim=-1) <= self.k_nbh_size
            mask = torch.zeros_like(vis_mask)
            mask[_msk] = vis_mask[_msk]

            # if there are less than 'nbh_size' unvisited nodes, correct mask
            # since we always need at least 'nbh_size' nodes for batching,
            # they will just be masked in the selection procedure
            missing_to_nbh_size = -mask.sum(-1) + self.k_nbh_size
            missing = missing_to_nbh_size > 0
            if missing.any():
                # create mask of the first 'missing_to_nbh_size' positions to set to true
                zmsk = ~mask[missing]
                zmsk = (
                    zmsk.cumsum(-1) == missing_to_nbh_size[missing, None]
                ).fliplr().cumsum(-1).fliplr().to(torch.bool)
                _msk = torch.zeros_like(mask)
                _msk[missing] = zmsk
                mask[_msk] = 1

            # select corresponding node indices
            select_idx = self.ordered_idx[mask].view(self.bs, -1)
            depot_nbh = depot_nbh[0].gather(dim=-1, index=select_idx)
            if depot_mask.all():
                return (
                    depot_nbh[:, None, :].expand(self.bs, self.max_concurrent_vehicles, -1) -
                    self.idx_inc[:, None, None]
                )

        # get other node nbh
        nbh = self.nbh_edges.view(2, self.bs, -1)[:, :, self.graph_size:]
        nbh = (
            nbh[0].view(self.bs, self.graph_size-1, self.k_nbh_size)
            # here we just clamp to enable the gather operation on dummy depot node indices,
            # they are then replaced below
            .gather(dim=1, index=(torch.clamp(node_idx-1, min=0))[:, :, None].expand(self.bs, -1, self.k_nbh_size))
        )

        if depot_mask.any():
            # replace depot nbh
            nbh[depot_mask] = depot_nbh.repeat_interleave(depot_mask.sum(-1), dim=0)

        return nbh - self.idx_inc[:, None, None]

    def compute_distance_matrix(self, coords: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Calculate (BS, N, N) distance (transit) matrix."""
        if normalize:
            return self._compute_normed_distance_matrix(coords, self.org_service_horizon)
        else:
            return dimacs_challenge_dist_fn(coords[:, :, None, :], coords[:, None, :, :])

    @staticmethod
    @torch.jit.script
    def _compute_normed_distance_matrix(coords: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return (
            dimacs_challenge_dist_fn(coords[:, :, None, :], coords[:, None, :, :]) /
            denom[:, None, None]
        )

    def _randint(self, bs: int, n: int, high: int, low: int = 0, replace: bool = False):
        """Draws n random integers between low (inc) and high (exc) for batch of size bs."""
        if self._one.device != self.device:
            self._one = self._one.to(device=self.device)
        return torch.multinomial(
            self._one[:, None, None].expand(-1, bs, high).view(bs, high),
            n, replacement=replace) + low

    def _get_nbh_and_mask(self) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """Returns the NBH of each node at which an active vehicle currently is positioned.
        Moreover, creates a feasibility mask over this NBH by
            - checking if node was already visited
            - checking if remaining capacity of vehicle is sufficient
            - checking if the node can still be served by the current vehicle within the respective TW

        Returns:
            nbh: (BS, max_concurrent, NBH),
            mask: (BS, max_concurrent, NBH)
        """
        # get node neighborhood of current nodes
        nbh = self.get_node_nbh(self.cur_node)

        # start creating mask (True where infeasible)
        # self-loops are masked automatically, since they just have been set as visited in step()
        # check which nodes were already visited
        mask = self._visited[:, None, :]\
            .expand(self.bs, self.max_concurrent_vehicles, -1)\
            .gather(dim=-1, index=nbh)

        # feasibility regarding capacity
        exceeds_cap = self.cur_cap[:, :, None] < self.demands[:, None, :]\
            .expand(self.bs, self.max_concurrent_vehicles, -1)\
            .gather(dim=-1, index=nbh)

        # feasibility regarding (hard) TWs
        # check if current time of vehicle + travel time to a node is smaller than
        # the latest start time of that node
        if self.inference:
            # select from distance matrix
            d = self.max_concurrent_vehicles * self.k_nbh_size
            t_delta = self._dist_mat[
                self._bidx[:, None].expand(-1, d).reshape(-1),
                self.cur_node[:, :, None]
                    .expand(self.bs, self.max_concurrent_vehicles, self.k_nbh_size)
                    .reshape(-1),
                nbh.view(-1)
            ].view(self.bs, self.max_concurrent_vehicles, -1)
        else:
            # compute on the fly
            idx_pair = torch.stack((
                self.cur_node[:, :, None].expand(self.bs, self.max_concurrent_vehicles, self.k_nbh_size).reshape(self.bs, -1),
                nbh.view(self.bs, -1)
            ), dim=-1).view(self.bs, -1)
            idx_coords = self.coords.gather(
                dim=1, index=idx_pair[:, :, None].expand(self.bs, -1, 2)
            ).view(self.bs, -1, 2, 2)
            t_delta = (
                    dimacs_challenge_dist_fn(idx_coords[:, :, 0, :], idx_coords[:, :, 1, :]) /
                    self.org_service_horizon[self._bidx][:, None]
            ).view(self.bs, self.max_concurrent_vehicles, -1)

        arrival_time = self.cur_time[:, :, None] + t_delta
        exceeds_tw = arrival_time > (
            self.tw[:, :, -1][:, None, :]
                .expand(self.bs, self.max_concurrent_vehicles, self.graph_size)
                .gather(dim=-1, index=nbh)
        )

        at_depot = self.cur_node == 0
        # debug checks
        if self.debug_lvl > 1:
            if (
                not (self._get_next_active_vehicle() == 0).any() and
                not self.visited.all(-1).any() and
                not self.inference
            ):
                assert self.cur_time[at_depot].sum() == 0
                assert (self.cur_cap[at_depot] - 1).sum() == 0
                assert self.cur_time_to_depot[at_depot].sum() == 0
                assert t_delta[at_depot][:, 0].sum() == 0
                assert exceeds_cap[at_depot].sum() == 0
                assert exceeds_tw[at_depot].sum() == 0

        # in case there is a vehicle starting from the depot
        # and there is at least one unvisited node and one unused vehicle left
        # then we mask the depot node
        mask_depot = (
            ~self.visited.all(-1) &
            ((self._finished.sum(-1) < self.max_concurrent_vehicles) | (self._get_next_active_vehicle() != 0))
        )
        mask_depot = at_depot & mask_depot[:, None].expand(-1, self.max_concurrent_vehicles)
        if mask_depot.any():
            mask[mask_depot, torch.zeros(mask_depot.sum(), dtype=torch.long, device=self.device)] = 1

        # combine masks
        mask = mask | exceeds_cap | exceeds_tw
        if (mask.all(-1)).any():
            raise RuntimeError(f"no feasible nodes left: {mask.all(-1).nonzero()}")
        return nbh, mask

    def _stack_to_tensor(self,
                         batch: List[RPInstance],
                         key: Union[str, int],
                         dtype: Optional[torch.dtype] = None
                         ) -> torch.Tensor:
        """Takes a list of instances and stacks the attribute
        indicated by 'key' into a torch.Tensor."""
        return torch.from_numpy(
            np.stack([x[key] for x in batch], axis=0)
        ).to(dtype=dtype if dtype is not None else self.fp_precision, device=self.device).contiguous()

    def _get_next_active_vehicle(self) -> torch.LongTensor:
        """
        Return the index of the next available vehicle.
        If no vehicles are available anymore, returns 0.
        """
        return torch.argmin((self._finished | self.active_vehicles).to(torch.int), dim=-1)

    def _get_observation(self) -> RPObs:
        """Gather the current observations."""
        nbh, mask = self._get_nbh_and_mask()
        return RPObs(
            batch_size=self.bs,
            node_features=torch.cat((
                self.coords,
                self.demands[:, :, None],
                self.tw,
                self.service_time[:, None, None].expand(self.bs, self.graph_size, 1),
                self.time_to_depot[:, :, None],
            ), dim=-1),
            node_nbh_edges=self.nbh_edges,
            node_nbh_weights=self.nbh_weights,
            # only select active
            tour_plan=self.tour_plan.gather(
                dim=1, index=self.active_to_plan_idx[:, :, None].expand(self.bs, -1, self.tour_plan.size(-1))
            ).to(dtype=torch.long),
            # these are just the features of currently active tours!
            tour_features=torch.cat((
                self.cur_node[:, :, None],  # just node idx!
                self.cur_cap[:, :, None],
                self.cur_time[:, :, None],
                self.cur_time_to_depot[:, :, None],
                # last entry encodes 'vehicle id' as well as 'number of remaining vehicles'
                ((-self.active_to_plan_idx + self.max_vehicle_number)/self.max_vehicle_number)[:, :, None],
            ), dim=-1),
            tour_edges=self.tour_edges,
            tour_weights=self.tour_weights,
            nbh=nbh,
            nbh_mask=mask,
        )

    def _update(self,
                tour_select: torch.LongTensor,
                next_node: torch.LongTensor,
                non_depot_mask: torch.BoolTensor,
                ) -> torch.Tensor:
        """Update tours."""
        previous_node = self.cur_node[self._bidx, tour_select]
        # update node
        self.cur_node[self._bidx, tour_select] = next_node

        # update load
        self.cur_cap[self._bidx, tour_select] = (
                self.cur_cap[self._bidx, tour_select] - self.demands[self._bidx, next_node]
        )
        if self.check_feasibility:
            assert (self.cur_cap[self._bidx, tour_select] >= 0).all()

        # update time
        if self.inference:
            # select from distance matrix
            cur_time_delta = self._dist_mat[self._bidx, previous_node, next_node]
        else:
            # compute on the fly
            idx_pair = torch.stack((previous_node, next_node), dim=0)
            idx_coords = self.coords[self._bidx, idx_pair]
            cur_time_delta = (
                    dimacs_challenge_dist_fn(idx_coords[0], idx_coords[1]) /
                    self.org_service_horizon[self._bidx]
            )

        tw = self.tw[self._bidx, next_node]
        arrival_time = self.cur_time[self._bidx, tour_select] + cur_time_delta
        if self.check_feasibility:
            if not (arrival_time <= tw[:, 1]).all():
                inf_msk = (arrival_time > tw[:, 1])
                td = arrival_time[inf_msk] - tw[inf_msk, 1]
                raise RuntimeError(f"arrival time exceeds TW "
                                   f"at idx: {inf_msk.nonzero()} w"
                                   f"ith time diff of {td}, "
                                   f"which equals {td/(1/self.org_service_horizon[inf_msk])} eps.")

        # add waiting time and service time for non-depot nodes
        cur_time_delta[non_depot_mask] = (
            cur_time_delta[non_depot_mask] +
            ((tw[:, 0] - arrival_time).clamp_(min=0) + self.service_time[self._bidx])[non_depot_mask]
        )
        self.cur_time[self._bidx, tour_select] = self.cur_time[self._bidx, tour_select] + cur_time_delta

        # update time to depot
        time_to_depot_delta = self.time_to_depot[self._bidx, next_node]
        previous_time_to_depot = self.cur_time_to_depot[self._bidx, tour_select]
        self.cur_time_to_depot[self._bidx, tour_select] = time_to_depot_delta

        # calculate cost
        cost = cur_time_delta + (time_to_depot_delta - previous_time_to_depot)

        return cost

    def _return_all(self) -> torch.Tensor:
        """Return all vehicles to depot and update corresponding buffers and cost."""
        must_return = (self.cur_node != self.depot_node[:, None])
        cost = self._zero
        if must_return.any():
            next_node = self.depot_node[:, None].expand(-1, self.max_concurrent_vehicles).clone()
            previous_node = self.cur_node.clone()
            # update node
            self.cur_node = next_node.clone()
            # update time
            if self.inference:
                # select from distance matrix
                cur_time_delta = self._dist_mat[
                    self._bidx[:, None].expand(self.bs, self.max_concurrent_vehicles).reshape(-1),
                    previous_node.view(-1),
                    next_node.view(-1)
                ].view(self.bs, self.max_concurrent_vehicles)
            else:
                # compute on the fly
                idx_pair = torch.stack((previous_node, next_node), dim=-1).view(self.bs, -1)
                idx_coords = self.coords.gather(
                    dim=1, index=idx_pair[:, :, None].expand(self.bs, -1, 2)
                ).view(self.bs, -1, 2, 2)
                cur_time_delta = (
                        dimacs_challenge_dist_fn(idx_coords[:, :, 0, :], idx_coords[:, :, 1, :]) /
                        self.org_service_horizon[self._bidx][:, None]
                ).view(self.bs, self.max_concurrent_vehicles)

            self.cur_time = self.cur_time + cur_time_delta
            if self.check_feasibility:
                assert (self.cur_time <= 1.0).all()
            # update time to depot
            self.cur_time_to_depot[:, :] = 0

            # calculate cost
            cost = cur_time_delta.sum(dim=-1)

        return cost.to(dtype=self.fp_precision)

    def _get_edges_to_render(self, b_idx: int = 0) -> List[np.ndarray]:
        """Select instance with b_idx and return corresponding edges as list of arrays."""
        # select all routes which are either finished or active (partial solutions)
        selection_mask = self._finished[b_idx] | self.active_vehicles[b_idx]
        tours = self.tour_plan[b_idx][selection_mask]
        max_pos = torch.argmin(tours, dim=-1)
        truncated_tours = []
        for tr, pos, fin in zip(tours, max_pos, self._finished[b_idx][selection_mask]):
            trtr = tr[:pos+1]
            if len(trtr) > 1:
                trtr = torch.cat((
                    torch.roll(trtr, shifts=1, dims=-1)[:, None],  # cyclic shift by 1
                    trtr[:, None]
                ), axis=1).T
                truncated_tours.append(trtr[:, :pos+fin].view(2, -1).cpu().numpy())
        return truncated_tours

    def _init_sampling(self):
        """Expand currently loaded instance(s) over sampling dimension."""
        bs, n, _ = self.coords.size()
        self.coords = self.coords[:, None, :, :].expand(bs, self._num_samples, n, 2).reshape(-1, n, 2).contiguous()
        self.demands = self.demands[:, None, :].expand(self.bs, self._num_samples, n).reshape(-1, n)
        self.tw = self.tw[:, None, :, :].expand(bs, self._num_samples, n, 2).reshape(-1, n, 2)
        self.service_time = self.service_time[:, None].expand(self.bs, self._num_samples).reshape(-1)
        self.time_to_depot = self.time_to_depot[:, None, :].expand(self.bs, self._num_samples, n).reshape(-1, n)
        self.org_service_horizon = self.org_service_horizon[:, None].expand(self.bs, self._num_samples).reshape(-1)
        if self.inference and self._dist_mat is not None:
            self._dist_mat = self._dist_mat[:, None, :, :].expand(self.bs, self._num_samples, n, n).reshape(-1, n, n)

        self.bs = int(self.bs * self._num_samples)
        self._bidx = torch.arange(self.bs, device=self.device)

    def _expand_sample_dimension(self, samples_fact: int):
        """Re-expand sampling dimension from imported selection to original sampling dimension."""
        cur_idx = torch.arange(0, self.num_samples, samples_fact, device=self.device)

        self._visited = self._visited[cur_idx].repeat_interleave(samples_fact, dim=0)
        self._finished = self._finished[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.tour_plan = self.tour_plan[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.active_vehicles = self.active_vehicles[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.next_index_in_tour = self.next_index_in_tour[cur_idx].repeat_interleave(samples_fact, dim=0)

        self.cur_node = self.cur_node[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.cur_cap = self.cur_cap[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.cur_time = self.cur_time[cur_idx].repeat_interleave(samples_fact, dim=0)
        self.cur_time_to_depot = self.cur_time_to_depot[cur_idx].repeat_interleave(samples_fact, dim=0)

        self._total = self._total.repeat_interleave(samples_fact, dim=0)

    def _reset_pomo(self):
        """Sample different starting nodes for POMO sample rollouts and
        directly move vehicles from depot to their start nodes."""
        # get the nodes in the vicinity of the depot
        nbh = self.get_node_nbh(self.depot_node)
        # remove depot self-loop
        nbh = nbh[:, :, 1:]
        
        if self.pomo_single_start_node:
            # select only one random start node per sample, independent of max_concurrent_vehicles
            num_start_nodes = 1
            # per convention we always select the node for the first tour
            start_tours = self._zero[None].expand(self.bs, -1).view(-1, 1)
            # order early TW by start time and select first num_samples
            nbh = nbh.view(-1, self.num_samples, self.max_concurrent_vehicles, self.k_nbh_size-1)[:, 0, 0, :]
            early_tw_idx = (
                self.tw.view(-1, self.num_samples, self.graph_size, 2)[:, 0, :, 0]
                    .gather(dim=-1, index=nbh).argsort(dim=-1)[:, :self.num_samples]
            )
            # all vehicles start at depot -> nbh is exactly the same for each sample of the same instance
            # so we get the first num_samples sorted start nodes of first sample and reshape back to batch
            start_nodes = nbh.gather(dim=-1, index=early_tw_idx).view(-1, 1)

        else:
            num_start_nodes = self.max_concurrent_vehicles
            # check the TW and select the fraction with the earliest TW as possible start nodes
            # since nodes with late TW cannot be optimal!
            k = math.floor((self.k_nbh_size-1) * self.pomo_nbh_frac)
            early_tw_idx = self.tw[:, :, 0].gather(dim=-1, index=nbh[:, 0, :]).argsort(dim=-1)[:, :k]
            # all vehicles start at depot -> nbh is exactly the same for all max_concurrent_vehicles per instance
            # so we get first only and expand to full max_concurrent_vehicles
            nbh = nbh.gather(dim=-1, index=early_tw_idx[:, None, :].expand(self.bs, self.max_concurrent_vehicles, k))

            # sample random idx
            # we need to sample num_samples different start node configurations of
            # the max_concurrent_vehicles used
            try:
                rnd_idx = self._randint(
                    bs=self.bs,
                    n=self.max_concurrent_vehicles,
                    high=k,
                )
            except RuntimeError:
                raise RuntimeError(f"sample set for POMO has size {k} which is too "
                                   f"small for {self.num_samples} samples! "
                                   f"Try increasing 'k_nbh_frac' of env.")

            start_nodes = nbh.gather(dim=-1, index=rnd_idx[:, :, None]).view(self.bs, -1)
            start_tours = torch.arange(self.max_concurrent_vehicles, device=self.device)[None, :].expand(self.bs, -1)

        # execute pseudo steps for each tour-start_node sample combination
        msk = torch.ones(self.bs, dtype=torch.bool, device=self.device)
        cost = 0
        for i in range(num_start_nodes):
            tour_select = start_tours[:, i]
            next_node = start_nodes[:, i]

            tour_plan_select = self.active_to_plan_idx[self._bidx, tour_select]

            cost += self._update(tour_select, next_node, msk)

            # add next node in tour
            nxt = self.next_index_in_tour[self._bidx, tour_select]
            self.tour_plan[self._bidx, tour_plan_select, nxt] = next_node.to(torch.int16)

            # increase idx of next position in tour
            self.next_index_in_tour[self._bidx, tour_select] = nxt + 1
            self._visited[self._bidx, next_node] = 1

        # depot node is never marked as visited!
        self._visited[:, 0] = 0

        # update graph data
        self.to_graph()

        self._total += cost
        self._step += num_start_nodes


# ============= #
# ### TEST #### #
# ============= #
def _test():
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS

    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    SMP = 128
    N = 100
    BS = 64
    BS_ = BS
    MAX_CON = 5
    CUDA = False
    SEED = 123
    POMO = True
    N_POMO = 8
    POMO_SINGLE = True
    if POMO:
        BS_ = BS//N_POMO
        SMP = 2 * BS_
    INFER = True

    device = torch.device("cuda" if CUDA else "cpu")

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(SEED)
    data = ds.sample(sample_size=SMP, graph_size=N)

    dl = DataLoader(
        data,
        batch_size=BS_,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(debug=True,
                device=device,
                max_concurrent_vehicles=MAX_CON,
                k_nbh_frac=0.4,
                pomo=POMO,
                pomo_single_start_node=POMO_SINGLE,
                num_samples=N_POMO,
                inference=INFER,
                )
    env.seed(SEED+1)

    for batch in dl:
        env.load_data(batch)
        obs = env.reset()
        done = False
        i = 0
        start_tws = env._stack_to_tensor(batch, "tw")[:, :, 1]
        #print(env.coords[:, 0])

        while not done:
            #print(i)
            # select tour randomly and then select available node with earliest TW
            tr = torch.randint(MAX_CON, size=(BS,), device=device)
            t_nbh = obs.nbh[torch.arange(BS), tr]
            t_msk = obs.nbh_mask[torch.arange(BS), tr]
            nd = torch.zeros(BS, dtype=torch.long, device=device)
            for j, (nbh, msk, start_tw) in enumerate(zip(t_nbh, t_msk, start_tws)):
                available_idx = nbh[~msk]   # mask is True where infeasible
                idx = available_idx[start_tw[available_idx].argsort(-1, descending=False)]
                nd[j] = idx[0]
            obs, rew, done, info = env.step(torch.stack((tr, nd), dim=-1))
            i += 1

        #print(info)
        sol = env.export_sol()
        print(sol)


def _test_io():
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS

    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    SMP = 64
    N = 50
    BS = 16
    BS_ = BS
    MAX_CON = 3
    CUDA = False
    SEED = 123
    POMO = True
    N_SMP = 16
    if N_SMP > 1:
        assert BS % N_SMP == 0
        BS_ = BS//N_SMP
        assert BS_ == 1
        SMP = 2 * BS_
    INFER = True    # IO only for inference!
    ITERS = 3
    NO_COST = True

    device = torch.device("cuda" if CUDA else "cpu")
    rnd = np.random.default_rng(SEED)

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(SEED)
    data = ds.sample(sample_size=SMP, graph_size=N)

    dl = DataLoader(
        data,
        batch_size=BS_,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(debug=True,
                device=device,
                max_concurrent_vehicles=MAX_CON,
                k_nbh_frac=0.4,
                pomo=POMO,
                num_samples=N_SMP,
                inference=INFER,
                )
    env.seed(SEED+1)

    for batch in dl:
        env.load_data(batch)
        obs = env.reset()
        start_tws = env._stack_to_tensor(batch, "tw")[:, :, 1]
        #print(env.coords[:, 0])

        for m in range(ITERS):
            print(f"iter: {m}")
            done = False
            i = 0
            while not done:
                #print(i)
                # select tour randomly and then select available node with earliest TW
                tr = torch.randint(MAX_CON, size=(BS,), device=device)
                t_nbh = obs.nbh[torch.arange(BS), tr]
                t_msk = obs.nbh_mask[torch.arange(BS), tr]
                nd = torch.zeros(BS, dtype=torch.long, device=device)
                for j, (nbh, msk, start_tw) in enumerate(zip(t_nbh, t_msk, start_tws)):
                    available_idx = nbh[~msk]   # mask is True where infeasible
                    idx = available_idx[start_tw[available_idx].argsort(-1, descending=False)]
                    nd[j] = idx[0]
                obs, rew, done, info = env.step(torch.stack((tr, nd), dim=-1))
                i += 1

            # export solution
            sol, cost = env.export_sol(3, 3, "dis")
            print(sol[0])

            # solution shape is (BS, num_samples, num_tours, ...)
            #assert len(sol) == BS
            new_solutions = []
            for s in sol:
                s = s[0]    # here only use first sample
                part = MAX_CON
                idx = (-np.array([len(t) for t in s])).argsort(axis=-1)
                part_idx = idx[:part]
                comp_idx = idx[part:]
                new_sol = []
                for idx in part_idx:
                    t = s[idx]
                    tlim = rnd.choice(np.arange(2, max(len(t)-1, 3)))
                    assert tlim >= 2
                    new_sol.append(t[:tlim])
                    #new_sol += [[el] for el in t[tlim:]]
                for idx in comp_idx:
                    new_sol.append([0] + s[idx] + [0])   # add depot idx at start and end
                new_solutions.append([new_sol[:28]])     # in list corresponding to num_samples=1

            if NO_COST:
                cost = None
            new_solutions = [s*N_SMP for s in new_solutions]
            obs = env.import_sol(new_solutions, cost)


def _profile():
    from torch.utils.data import DataLoader
    from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS
    from torch.profiler import profile, record_function, ProfilerActivity, schedule

    SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
    LPATH = "./solomon_stats.pkl"
    SMP = 128
    N = 50
    BS = 64
    BS_ = BS
    MAX_CON = 3
    CUDA = False
    SEED = 123
    POMO = False
    N_POMO = 8
    if POMO:
        BS_ = BS // N_POMO
        SMP = 2*BS_
    INFER = True

    device = torch.device("cuda" if CUDA else "cpu")

    ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
    ds.seed(SEED)
    data = ds.sample(sample_size=SMP, graph_size=N)

    dl = DataLoader(
        data,
        batch_size=BS_,
        collate_fn=lambda x: x,  # identity -> returning simple list of instances
        shuffle=False
    )

    env = RPEnv(debug=True,
                device=device,
                max_concurrent_vehicles=MAX_CON,
                k_nbh_frac=0.4,
                pomo=POMO,
                num_samples=N_POMO,
                inference=INFER,
                )
    env.seed(SEED + 1)

    s = schedule(
        wait=0,
        warmup=0,
        active=N,
        repeat=2
    )

    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if CUDA else []),
        schedule=s,
        record_shapes=False,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for batch in dl:
            with record_function(">>env.load()"):
                env.load_data(batch)
            with record_function(">>env.reset()"):
                obs = env.reset()
            done = False
            i = 0
            start_tws = env._stack_to_tensor(batch, "tw")[:, :, 1]
            # print(env.coords[:, 0])

            while not done:
                # print(i)
                # select tour randomly and then select available node with earliest TW
                tr = torch.randint(MAX_CON, size=(BS,), device=device)
                t_nbh = obs.nbh[torch.arange(BS), tr]
                t_msk = obs.nbh_mask[torch.arange(BS), tr]
                nd = torch.zeros(BS, dtype=torch.long, device=device)
                for j, (nbh, msk, start_tw) in enumerate(zip(t_nbh, t_msk, start_tws)):
                    available_idx = nbh[~msk]  # mask is True where infeasible
                    idx = available_idx[start_tw[available_idx].argsort(-1, descending=False)]
                    nd[j] = idx[0]
                with record_function(">>env.step()"):
                    obs, rew, done, info = env.step(torch.stack((tr, nd), dim=-1))

                i += 1

            with record_function(">>env.export_sol()"):
                sol = env.export_sol()

    # report
    print(prof.key_averages(group_by_stack_n=7).table(
        sort_by="gpu_time_total" if CUDA else "cpu_time_total",
        row_limit=100))
