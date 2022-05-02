#
from typing import List, Dict, Tuple, Union, Optional
import logging
from warnings import warn
import numpy as np
import time
import sys

from lib.utils.or_utils import (
    add_timewindows_custom,
    parse_assignment,
    is_feasible
)
from lib.utils.challenge_utils import dimacs_challenge_dist_fn_np

# init ortools logger to disable output to STDERR
from ortools.init import pywrapinit

pywrapinit.CppBridge.InitLogging("gort")
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

logger = logging.getLogger(__name__)

STATUS_MAP = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_FAIL",
    3: "ROUTING_FAIL_TIMEOUT",
    4: "ROUTING_INVALID",
}


def dimacs_challenge_dist(i: Union[np.ndarray, float],
                          j: Union[np.ndarray, float]
                          ) -> np.ndarray:
    """
    times/distances are obtained from the location coordinates,
    by computing the Euclidean distances truncated to one
    decimal place:
    $d_{ij} = \frac{\floor{10e_{ij}}}{10}$
    where $e_{ij}$ is the Euclidean distance between locations i and j
    """
    return np.floor(10 * np.sqrt(((i - j) ** 2).sum(axis=-1))) / 10


def time_matrix(X, service_time):
    all_dists = []
    for i in range(len(X)):
        np_dists = dimacs_challenge_dist(X, X[i])
        all_dists.append(np_dists)
    t_mat = np.stack(all_dists, axis=0)
    transit_mat = t_mat + service_time.reshape(101, 1)
    np.fill_diagonal(transit_mat, 0)
    return transit_mat


class LocalSearch:
    """
    Local Search for the CVRP-TW via Google ORTools.
    It takes S samples of solution tour plans and
    outputs S improved samples of those solution plans.

    The functionality is heavily based on the source code
    provided by the Google ORTools API:
    https://developers.google.com/optimization/
    """

    def __init__(self,
                 check_feasibility: bool = False,
                 search_strategy: str = 'gls',
                 search_timelimit: int = 100,
                 max_time_wo_improvement: Union[int, float] = 0,
                 penalty_v: int = 10000,
                 globalspan: int = 100,
                 precision: int = 100,
                 verbose: bool = False,
                 raise_: bool = True,
                 **kwargs):
        """

        Args:
            check_feasibility: check feasibility of solution samples
            search_strategy: which ORT search to implement
            search_timelimit: for meta-heuristic
            max_time_wo_improvement: max time limit to continue search without improvement (0 to disable)
            penalty_v: penalty for using using more vehicles than allowed
            globalspan: cost for minimizing globalspan of tour
            precision: magnitude of floating point precision equivalent to use
        """
        self.check_feasibility = check_feasibility
        self.search_strategy = search_strategy
        self.search_timelimit = search_timelimit
        self.max_time_wo_improvement = max_time_wo_improvement
        self.penalty_v = penalty_v
        self.globalspan = globalspan
        self.precision = precision
        self.verbose = verbose
        self.raise_ = raise_

        self.static_properties = None

    def load_instance(self, instance: Dict, dist_mat: Optional[Union[List, np.ndarray]] = None):
        """
        The main instance properties remain static.
        For each sample solution the distance matrix etc. do not change.
        """
        features = instance['features']
        if dist_mat is None:
            coords = np.stack((features['x_coord'],
                               features['y_coord']), axis=-1) / 100  # since dist_fn takes *100
            dist_mat = (
                    dimacs_challenge_dist_fn_np(coords[:, None, :], coords[None, :, :]) +
                    features['service_time'].to_numpy()[:, None]
            )
            np.fill_diagonal(dist_mat, 0)

        self.static_properties = {
            'depot': 0,
            'dist_mat': (dist_mat * self.precision).astype(int),  # includes service time
            'cap': int(instance['vehicle_capacity']),
            'vehicle_max': int(instance['max_vehicle_number']),
            'demands': features['demand'].astype(int).tolist(),  # includes 0 for depot demand
            'time_windows': (
                    np.stack((features['tw_start'], features['tw_end']), axis=-1) * self.precision
            ).astype(int).tolist()
        }

    def search(self, cur_sol):
        """runs the search for a single solution of the problem"""

        assert self.static_properties is not None, f"need to load instance first!"

        # check feasibility of current solution (only one solution)
        if self.check_feasibility:
            assert is_feasible(cur_sol, self.static_properties, verbose=self.verbose)

        #print('cur_sol', cur_sol)
        num_vehicles = len(cur_sol)
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(self.static_properties['dist_mat']),
            num_vehicles,
            self.static_properties['depot']
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Routing Solver Object to access current improvements
        solver = routing.solver()

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.static_properties['dist_mat'][from_node, to_node]

        # transit measured in distance or time.
        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # for CVRP-TW - create callbacks for capacity and TW constraints
        # 1. Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return self.static_properties['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.static_properties['cap']] * num_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # 2. Add Time window Constraint
        add_timewindows_custom(
            routing, manager, self.static_properties, transit_callback_index,
            num_vehicles=num_vehicles, globalspan_cost=self.globalspan, scale=self.precision
        )

        # Set penalty for using more vehicles than allowed
        if num_vehicles > self.static_properties['vehicle_max']:
            for i in range(self.static_properties['vehicle_max'], num_vehicles):
                routing.SetFixedCostOfVehicle(self.penalty_v, i)

        # To minimize Total Time (incl. wait times) - set Span Cost Coeff
        # pywrapcp.RoutingDimension.SetSpanCostCoefficientForAllVehicles(1000)

        # Set default search parameters.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # Only have default search + time limit:
        if self.search_strategy == 'GORT_default':
            # default is set - only set timelimit for search
            search_parameters.time_limit.seconds = self.search_timelimit
        # Set Additional Meta-Search parameters:
        elif self.search_strategy == 'gls':
            logger.info(f'performing GLS with search time limit of: {self.search_timelimit}')
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.seconds = self.search_timelimit
            # search_parameters.solution_limit = 100
            search_parameters.log_search = True
        elif self.search_strategy == 'greedy_descent':
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
            search_parameters.time_limit.seconds = self.search_timelimit
            search_parameters.log_search = True
        else:
            # if unknown search strategy specified do GORT default search
            logger.info('Search Strategy not known or not implemented, using GORT defaults')
            search_parameters.time_limit.seconds = self.search_timelimit

        ## additional search parameter settings != Default:
        # search_parameters.use_cp_sat = 3
        # search_parameters.use_depth_first_search = True
        # search_parameters.relocate_expensive_chain_num_arcs_to_consider = 25  # default seems 20
        # search_parameters.guided_local_search_lambda_coefficient = 7  # default 5
        # multi_armed_bandit_compound_operator_exploration_coefficient = 50  # default 43
        # operators
        search_parameters.local_search_operators.use_extended_swap_active = 3  # 'BOOL_TRUE'
        search_parameters.local_search_operators.use_relocate_neighbors = 3
        search_parameters.local_search_operators.use_cross_exchange = 3
        # search_parameters.local_search_operators.use_relocate_and_make_active = 3

        # Add search cfg and close model. We need to do this here since
        # routing.ReadAssignmentFromRoutes quietly closes the model for any further changes
        # -> https://github.com/google/or-tools/issues/1841
        routing.CloseModelWithParameters(search_parameters)

        if self.max_time_wo_improvement > 0:
            # Limit search if no improvement:
            bestCollector = solver.BestValueSolutionCollector(False)
            bestCollector.AddObjective(routing.CostVar())
            routing.AddSearchMonitor(bestCollector)
            bestSolution = dict(score=sys.maxsize, clock=0, startTime=time.time())

            def stop_search():
                """Callback checking improvement vs. time and stopping search
                if there has been no improvement for max_time_wo_improvement seconds."""
                if bestCollector.SolutionCount() > 0 and bestCollector.ObjectiveValue(0) < bestSolution.score:
                    bestSolution['score'] = bestCollector.ObjectiveValue(0)
                    bestSolution['clock'] = bestCollector.WallTime(0)
                    # print(bestSolution.solution(0))  # that's the solution
                    # stop search?
                    return (
                            bestSolution['score'] < sys.maxsize and
                            time.time() - bestSolution['startTime'] > bestSolution['clock'] * 0.001 +
                            self.max_time_wo_improvement
                    )

            routing.AddSearchMonitor(solver.CustomLimit(stop_search))

        # set current solution as initial solution
        initial_solution = routing.ReadAssignmentFromRoutes(cur_sol, True)
        if initial_solution is None:
            logger.error(f"Routing status: {STATUS_MAP[routing.status()]}")
            if self.raise_:
                raise RuntimeError(f"provided initial solution is not feasible.")
            else:
                warn(f"RuntimeError due to Solution not feasible, Return current Solution")
                return cur_sol, None, None

        else:
            # Solve the problem i.e. Improve initial solution
            solution = routing.SolveFromAssignmentWithParameters(initial_solution,
                                                                 search_parameters)
            status = STATUS_MAP[routing.status()]
            logger.info(f"Routing status: {status}")

            if solution:
                routes, total_cost, info = parse_assignment(
                    self.static_properties,
                    manager, routing, solution,
                    num_vehicles=num_vehicles
                )
                cost = total_cost / self.precision
            else:
                logger.info('No Improved solution')
                routes = cur_sol
                cost, info = None, None

            return routes, cost, info
