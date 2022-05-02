#
from itertools import tee
from typing import List, Dict, Tuple, Union, Optional
from warnings import warn
import numpy as np

TIME = "Time"


def parse_assignment(
        data: Dict,
        manager,
        routing,
        solution: List[List],
        num_vehicles: int,
        warn_: bool = True,
        **kwargs
):
    transit_dim = routing.GetDimensionOrDie(TIME)
    total_cost = 0
    routes = []
    info = {}
    for vehicle_id in range(num_vehicles):
        cust_ids = []
        load_lst = []
        wait_tm_lst = [0]

        index = routing.Start(vehicle_id)
        cur_time = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            cust_ids.append(node_index)
            load_lst.append(data['demands'][node_index])

            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            cur_time += data['dist_mat'][node_index, next_node]
            wait_tm = max(0, data['time_windows'][next_node][0] - cur_time)
            cur_time += wait_tm
            wait_tm_lst.append(wait_tm)

        # add from last index
        time_var = transit_dim.CumulVar(index)
        cost = solution.Min(time_var)
        cust_ids.append(manager.IndexToNode(index))
        load_lst.append(0)  # depot
        if cost != cur_time:
            if warn_:
                warn(f"inconsistencies during cost and time calculations: "
                     f"\n   cost {cost} != time {cur_time} !")
        total_cost += cost

        if len(cust_ids) > 2:
            routes.append(cust_ids)
            info[vehicle_id] = {
                "route": cust_ids,
                "loads": load_lst,
                "waiting_time": wait_tm_lst,
                "cost": cost
            }
    return routes, total_cost, info


def is_feasible(solution, features, verbose=False, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(solution)

    # max vehicle check:
    if len(solution) > features['vehicle_max']:
        warn(f"num_tours > max vehicles!")

    # check capacity constraint:
    cap = features['cap']
    for t_idx, tour in enumerate(solution[start:end]):
        tour_demand = 0
        for i, node in enumerate(tour):
            tour_demand += features['demands'][node]
            if tour_demand > cap:
                raise RuntimeError(f"Capacity constraint violated in tour={t_idx} at {tour[i-1]}->{node}:"
                                   f"\n     demand {tour_demand} > cap {cap}!"
                                   f"\n     tour: {tour},"
                                   f"\n     tour_demands: {[features['demands'][idx] for idx in tour]}")

    # check time window constraint
    problems = []
    for t_idx, tour in enumerate(solution[start:end]):
        problem_no = 0
        no_problem = 0
        time_acc = features['dist_mat'][0][tour[0]]
        # wait_t: 0 --> start node
        time_acc += max(0, features['time_windows'][tour[0]][0]-time_acc)
        if verbose:
            print(f"\n___ TOUR: {t_idx} __________________")
            print('depot outgoing start time:', time_acc)
            print('time_window start node:', features['time_windows'][tour[0]])
        tour_ = tour.copy()
        tour_.extend([0])
        for i, j in pairwise(tour_):
            travel_t = features['dist_mat'][i][j]
            t_travel_t = time_acc + features['dist_mat'][i][j]
            time_w_j = features['time_windows'][j]
            time_acc += features['dist_mat'][i][j]
            if verbose:
                print(f'\ni->j: {i}->{j}')
                print('travel_t i,j', travel_t)
                print('travel_t + acc_t', t_travel_t)
                print('time window j', time_w_j)
                print('updated acc_time', time_acc)
            # check if end of tw is reached:
            if time_acc > features['time_windows'][j][1]:
                print(">>INFEASIBLE<<")
                print(f"tour: {t_idx}, i->j: {i}->{j}")
                problem_no += 1
            else:
                no_problem += 1

            # add waiting time to time_acc:
            if verbose:
                print('wait_time:', features['time_windows'][j][0]-time_acc)
            time_acc += max(0, features['time_windows'][j][0]-time_acc)

        problems.append(problem_no)

    if (np.array(problems) == 0).all():
        return True
    else:
        print('infeasibility encountered:')
        print(np.array(problems))
        return False


def add_timewindows_custom(
        routing,
        manager,
        data,
        transit_callback_index,
        num_vehicles,
        globalspan_cost: int = 100,
        scale: int = 100,
):
    # Add Time Windows constraint.
    routing.AddDimension(
        transit_callback_index,
        5000000000, # int(1000*scale),  # allow waiting time
        9000000000, # int(10000*scale),  # maximum time per vehicle
        True,  # force start cumul to zero.
        TIME)
    time_dimension = routing.GetDimensionOrDie(TIME)
    time_dimension.SetGlobalSpanCostCoefficient(globalspan_cost)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def total_cost_transit_r(sol_list, x_dist):
    dist_route = 0
    for k, l in pairwise(sol_list):
        # print('k,l',(k,l))
        dist_route += x_dist[k, l]

    return dist_route


def get_search_sol(data, manager, routing, solution, num_vehicles):
    # print(f'Objective: {solution.ObjectiveValue()}')
    time_dimension = routing.GetDimensionOrDie(TIME)
    total_time = 0
    total_load = 0
    route_plan = {}

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_load = 0
        info_v = []
        plan_v = []
        cust_ids = []
        load_lst = []
        wait_t_r = []
        curr_time = 0
        ####################################
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            # print('\nnode_index', node_index)
            route_load += data['demands'][node_index]
            time_var = time_dimension.CumulVar(index)
            wait_t = max(0, solution.Min(time_var)-curr_time)
            cust_ids.append(node_index)
            load_lst.append(route_load)
            wait_t_r.append(wait_t)
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            curr_time += data['dist_mat'][node_index, next_node]

        time_var = time_dimension.CumulVar(index)
        wait_t = solution.Min(time_var) - curr_time
        wait_t_r.append(wait_t)

        cust_ids.append(manager.IndexToNode(index))
        load_lst.append(route_load)
        total_time += solution.Min(time_var)

        plan_v.append(cust_ids)
        plan_v.append(load_lst)
        plan_v.append(wait_t_r[1:])

        total_load += route_load
        info_v.append(plan_v)
        info_v.append(solution.Min(time_var))
        route_plan[vehicle_id] = info_v
        ##################

    route_plan['total_time'] = total_time
    return route_plan


def get_list_sol(sol_dct):
    sol_routes = []
    route_info = []
    for v in range(0, len(sol_dct.keys()) - 1):
        # print(v)
        if len(sol_dct[v][0][0]) > 2:
            sol_routes.append(sol_dct[v][0][0])
            capa = sol_dct[v][0][1][-1]
            # print('remain_capa', capa)
            route_info.append({
                'wait_times': sol_dct[v][0][2],
                'utilized_capa': sol_dct[v][0][1][-1],
                'route_duration': sol_dct[v][1]
            })

    return sol_routes, route_info, sol_dct['total_time']


# ref: https://www.geeksforgeeks.org/python-split-list-into-lists-by-particular-value/
def list_of_lists(list_, with_zeros=True):
    if with_zeros:
        size = len(list_)
        idx_list = [idx + 1 for idx, val in
                    enumerate(list_[1:]) if val == 0]
        # print(idx_list)
        res = [list_[i: j] + [0] for i, j in
               zip([0] + idx_list, idx_list +
                   ([size] if idx_list[-1] != size else []))]
        return res
    else:
        size = len(list_)
        idx_list = [idx + 1 for idx, val in
                    enumerate(list_[1:]) if val == 0]
        # print(idx_list)
        res = [list_[(i + 1): j] for i, j in
               zip([0] + idx_list, idx_list +
                   ([size] if idx_list[-1] != size else []))]
        return res


def get_ck(tour, dist):
    c_ks = [-1]
    tour = np.append(tour, 0)
    for idx, k in enumerate(tour[1:-1]):
        if not k == 0:
            c_k = dist[tour[idx], k] + dist[k, tour[idx + 2]]
            c_ks.append(c_k)
        else:
            c_ks.append(-1)
    return np.array(c_ks)
