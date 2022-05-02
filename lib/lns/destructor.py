#
from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np

from lib.utils.seq_match import rm_common_subsequences


class Destructor:
    """Class handling different destruction methods for LNS."""
    RM_P_MODES = ["random_nodes", "random_cut", "const_cut", "wait_time_cut"]

    def __init__(self,
                 num_partial: int,
                 destruction_mode: str = "random_from_route",
                 rm_partial_mode: str = "random_cut",
                 rm_complete_mode: str = "random",
                 frac_rm_nodes_from_partial: float = 0.0,
                 frac_rm_complete_routes: float = 0.0,
                 similarity_threshold: float = 0.5,
                 seed: int = 1,
                 ):
        """

        Args:
            num_partial: max number of partial routes
            destruction_mode: id of destruction method to use
                              one of ["random_from_route", ]
            rm_partial_mode: method to use for partial route destruction
                             one of ["random_nodes", "random_cut", "const_cut", "wait_time_cut"]
                             or "random" to randomly use one of the above each iteration
            rm_complete_mode: method to use for complete removal of tours
                              one of ["random", "smallest", "similarity", "waiting_time"]
            frac_rm_nodes_from_partial: optional fraction of nodes to remove from partial tours
            frac_rm_complete_routes: optional fraction of complete routes to remove
            similarity_threshold: threshold for similarity score when removing similar routes
            seed: seed for random generator
        """
        self.num_partial = num_partial
        self.destruction_mode = destruction_mode.lower()
        self.rm_partial_mode = rm_partial_mode.lower()
        self.rm_complete_mode = rm_complete_mode.lower()
        if self.rm_complete_mode in ["similarity", "waiting_time"]:
            assert self.rm_partial_mode != "wait_time_cut",\
                f"rm_partial_mode='wait_time_cut' does not work with " \
                f"rm_complete_mode='similarity' or 'waiting_time'"
            self.rm_p_modes = self.RM_P_MODES[:-1].copy()   # remove wait_time_cut from list
        else:
            self.rm_p_modes = self.RM_P_MODES.copy()
        assert frac_rm_nodes_from_partial < 1
        self.frac_rm_nodes_from_partial = frac_rm_nodes_from_partial
        assert frac_rm_complete_routes < 1
        self.frac_rm_complete_routes = frac_rm_complete_routes
        self.similarity_threshold = similarity_threshold
        #
        self.rnd = np.random.default_rng(seed)

    def rnd_int_1(self, n: int):
        """Return a random int x for cut or selection (1 <= x < n)."""
        if n <= 2:
            return 1
        else:
            return self.rnd.integers(1, n-1, (1,))[0]

    def destruct(self,
                 solutions: List[List[List]],
                 costs: Optional[List[float]] = None,
                 infos: Optional[List[Dict]] = None,
                 dist_mat: Optional[List[np.ndarray]] = None,
                 **kwargs) -> List[List[List]]:
        """Destruct routes in solutions according to configuration."""
        if self.rm_complete_mode == "similarity":
            solutions, costs = self.rm_similar_routes(solutions, costs, **kwargs)
        elif self.rm_complete_mode == "waiting_time":
            solutions, costs = self.rm_routes_by_wait_time(solutions, infos, **kwargs)
        mutilated = []
        for i, sol in enumerate(solutions):
            if self.destruction_mode == "random_from_route":
                info = infos[i] if infos is not None else None
                sol = self.random_from_route(sol, info)
            else:
                raise ValueError(f"unknown destruction mode: '{self.destruction_mode}'")
            mutilated.append(sol)
        return mutilated
    
    def random_from_route(self, sol: List[List], infos: Optional[Dict] = None) -> List[List]:
        """Destroy a random part of maximum num_partial tours."""
        lens = np.array([len(t) for t in sol])
        # check if the solutions comes with depot idx 0 added
        # only destruct tours with more than 1 node apart from the depot
        tour = sol[0]
        if tour[0] == tour[-1] == 0:
            max_idx = 3  # singleton tour + start and end at depot
        else:
            max_idx = 1  # just singleton tour
            
        candidate_idx = (lens > max_idx).nonzero()[0]
        candidate_tours = self.rnd.choice(candidate_idx, self.num_partial, replace=False)
        c_tours, p_tours = [], []
        for i in range(len(sol)):
            tour = sol[i]
            if i in candidate_tours:
                info = infos.get(i) if infos is not None else None
                # destruct part of tour
                p_tours.append(self.create_partial_route(tour, info=info))
            else:
                # check if tour is a singleton tour, which is ignored in that case
                # (we destruct all complete singleton tours by default)
                if len(tour) > max_idx:
                    # check depot idx and add if missing
                    if not (tour[0] == tour[-1] == 0):
                        tour = [0] + tour + [0]
                    c_tours.append(tour)

        # remove some of the complete tours if specified
        c_tours = self.rm_complete_routes(c_tours)

        return c_tours + p_tours

    def create_partial_route(self, tour: List, **kwargs) -> List[List]:
        assert self.num_partial > 0
        if tour[0] == tour[-1] == 0:  # remove depot idx if included in tour
            tour = tour[1:-1]
        if self.rm_partial_mode == "random":
            # each iter randomly select a destruction operator
            mode = self.rnd.choice(self.rm_p_modes, 1)
        else:
            mode = self.rm_partial_mode
        if mode == "random_nodes":
            return self._rm_random_nodes(tour)
        elif mode == "random_cut":
            return self._random_cut(tour)
        elif mode == "const_cut":
            assert self.frac_rm_nodes_from_partial > 0, \
                f"for const_cut must specify 'frac_rm_nodes_from_partial' > 0."
            return self._const_cut(tour)
        elif mode == "wait_time_cut":
            return self._wait_time_cut(tour, **kwargs)
        else:
            raise ValueError(f"unknown partial route removal mode: '{mode}'")

    def _rm_random_nodes(self, tour: List) -> List:
        """Remove random nodes from the provided tour."""
        n = len(tour)
        if self.frac_rm_nodes_from_partial > 0:
            n_rm = int(max(round(self.frac_rm_nodes_from_partial * n), 1))
        else:
            n_rm = self.rnd_int_1(n)
        rm_idx = self.rnd.choice(np.arange(n), size=min(n_rm, n), replace=False)
        return [e for i, e in enumerate(tour) if i not in rm_idx]

    def _random_cut(self, tour: List) -> List:
        """Cut route at random idx."""
        n = len(tour)
        cut_idx = self.rnd_int_1(n)
        return tour[:cut_idx]

    def _const_cut(self, tour: List) -> List:
        """Cut num_rm_nodes_from_partial last nodes from route."""
        # remove at least 1 node
        cut_idx = int(max(round(self.frac_rm_nodes_from_partial*len(tour)), 1))
        return tour[:-cut_idx]

    def _wait_time_cut(self, tour: List, info: Dict):
        """Cut the tour before the node with the highest waiting time."""
        if info is None:
            # fallback in case no infos are available
            return self._random_cut(tour)
        wtt = info['waiting_time']
        if len(tour) == len(wtt)-2:
            wtt = wtt[1:-1]     # remove depot times
        cut_idx = np.argmax(wtt)
        return tour[:cut_idx]

    def rm_complete_routes(self, tours: List[List]) -> List[List]:
        if self.frac_rm_complete_routes > 0:
            if self.rm_complete_mode == "random":
                return self._rm_random_routes(tours)
            elif self.rm_complete_mode == "smallest":
                return self._rm_smallest_routes(tours)
            elif self.rm_complete_mode in ["similarity", "waiting_time"]:
                pass    # done on solution pairs before call to this function
            else:
                raise ValueError(f"unknown complete route removal mode: "
                                 f"'{self.rm_complete_mode}'")
        return tours

    def _rm_random_routes(self, tours: List[List]) -> List[List]:
        """Removes 'num_rm_complete_routes' random tours from provided list."""
        n_c = len(tours)
        n_rm = int(max(round(self.frac_rm_complete_routes*n_c), 1))
        rm_idx = self.rnd.choice(np.arange(n_c), size=n_rm, replace=False)
        return [ct for i, ct in enumerate(tours) if i not in rm_idx]

    def _rm_smallest_routes(self, tours: List[List]) -> List[List]:
        """Removes the 'num_rm_complete_routes' smallest tours
        (with smallest number of nodes) from provided list."""
        n_c = len(tours)
        n_rm = int(max(round(self.frac_rm_complete_routes*n_c), 1))
        size = np.array([len(t) for t in tours])
        rm_idx = np.argsort(size)[:n_rm]
        return [ct for i, ct in enumerate(tours) if i not in rm_idx]

    def rm_similar_routes(self,
                          plans: List[List[List]],
                          costs: Optional[List[float]],
                          **kwargs) -> Tuple[List[List[List]], Any]:
        """Select the best plans and then remove the routes in all other plans
        which are the most similar to the routes in the best plans."""
        assert costs is not None
        assert self.frac_rm_complete_routes > 0
        assert len(plans) == len(costs)
        num = len(costs)
        num_best = max(np.sqrt(num).astype(int), min(2, num))
        bst = np.argsort(costs)[:num_best]
        best_plans = [p for i, p in enumerate(plans) if i in bst]
        candidate_plans = [p for i, p in enumerate(plans) if i not in bst]
        steps = np.linspace(0, num, num_best+1, dtype=int)
        new_plans = []
        for b_plan, s, e in zip(best_plans, steps[:-1], steps[1:]):
            for c_plan in candidate_plans[s:e]:
                n_rm = int(max(round(self.frac_rm_complete_routes * len(c_plan)), 1))
                new_plans.append(
                    rm_common_subsequences(
                        b_plan, c_plan,
                        k=n_rm,
                        tau=self.similarity_threshold
                    )
                )
        #
        return best_plans + new_plans, None

    def rm_routes_by_wait_time(self,
                               plans: List[List[List]],
                               infos: List[Dict],
                               **kwargs) -> Tuple[List[List[List]], Any]:
        """Remove from each tour plan the tours which accumulate the highest waiting time."""
        assert self.frac_rm_complete_routes > 0
        assert infos is not None, f"route removal based on waiting time requires info dict."
        assert len(plans) == len(infos)
        new_plans = []
        for plan, p_info in zip(plans, infos):
            if len(plan) != len(p_info):
                # fallback to random removal
                new_plans.append(self._rm_random_routes(plan))
            else:
                n_rm = int(max(round(self.frac_rm_complete_routes * len(plan)), 1))
                wtt = [sum(d['waiting_time']) for d in p_info.values()]
                rm_idx = np.argsort(-np.array(wtt))[:n_rm]
                new_plans.append([t for i, t in enumerate(plan) if i not in rm_idx])

        return new_plans, None
