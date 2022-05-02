#
import os
import io
from warnings import warn
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as stats
from sklearn.cluster import KMeans

from lib.routing.formats import RPInstance

__all__ = [
    'read_tsplib_cvrptw',
    'normalize_instance',
    'to_rp_instance',
    'load_instance',
    'process_group',
]


def read_tsplib_cvrptw(pth: str):
    """
    Read a CVRP-TW instance in TSPLIB95 format.
    Parts adapted from https://github.com/iRB-Lab/py-ga-VRPTW/blob/master/gavrptw/utils.py
    """
    instance = {}
    feature_names = ['node_id', 'x_coord', 'y_coord', 'demand', 'tw_start', 'tw_end', 'service_time']
    features = []
    with io.open(pth, 'rt', newline='') as f:
        for i, line in enumerate(f, start=1):
            if i in [2, 3, 4, 6, 7, 8, 9]:
                # description strings
                pass
            elif i == 1:
                # <Instance name>
                instance['id'] = line.strip()
            elif i == 5:
                # <Maximum vehicle number>, <Vehicle capacity>
                values = line.strip().split()
                instance['max_vehicle_number'] = int(values[0])
                instance['vehicle_capacity'] = float(values[1])
            else:
                # <Customer number>, <X coord>, <Y coord>, <Demand>, <Ready time>, <Due date>, <Service time>
                values = line.strip().split()
                assert len(values) == 7
                features.append([float(v) for v in values])

        # depot is first line (idx 0) of features
        instance['depot_idx'] = 0
        # convert features into DataFrame
        df = pd.DataFrame(data=features, columns=feature_names)
        df.set_index('node_id')
        df.drop(labels='node_id', axis=1, inplace=True)
        # add length of TW
        df['tw_len'] = df.tw_end-df.tw_start
        instance['features'] = df

    return instance


def normalize_instance(instance):
    df = deepcopy(instance['features'])
    # x/y coords are integers in [0, 100]
    df['x_coord'] /= 100
    df['y_coord'] /= 100
    # demand will be normalized by vehicle capacity
    df['demand'] /= instance['vehicle_capacity']
    # TW and service times are normalized by service horizon (TW of depot)
    service_horizon = df['tw_end'][0]
    df['tw_start'] /= service_horizon
    df['tw_end'] /= service_horizon
    df['service_time'] /= service_horizon
    df['tw_len'] /= service_horizon

    # to calculate the correct distance matrix of normalized coordinates interpreted as times
    # for constraint checking, one needs to correct for the normalization
    #instance['dist_to_time_factor'] = 100 / service_horizon
    instance['org_service_horizon'] = service_horizon
    instance['norm_features'] = df

    return instance


def to_rp_instance(instance) -> RPInstance:
    """Convert loaded and pre-processed TSPLIB instance to RPInstance."""
    df = instance['norm_features']
    coords = df.loc[:, ('x_coord', 'y_coord')].to_numpy()
    demand = df.loc[:, 'demand'].to_numpy()
    tw = df.loc[:, ('tw_start', 'tw_end')].to_numpy()
    service_time = df.loc[:, 'service_time'].to_numpy()
    assert np.all(service_time[1:] == service_time[1])

    # infer type
    # for Solomon and Gehring&Homberger instances can be inferred from capacity
    if instance['vehicle_capacity'] < 500:
        type = "1"
    else:
        type = "2"
    # infer TW fraction
    org_df = instance['features'].loc[1:, :]  # without depot!
    has_tw = (org_df.tw_start != 0)
    tw_frac = has_tw.sum() / org_df.shape[0]

    return RPInstance(
        coords=coords,
        demands=demand,
        tw=tw,
        service_time=service_time[1],
        graph_size=coords.shape[0],
        org_service_horizon=instance['org_service_horizon'],
        max_vehicle_number=instance['max_vehicle_number'],
        vehicle_capacity=1.0,  # is normalized
        service_horizon=1.0,  # is normalized
        depot_idx=[0],
        type=str(type),
        tw_frac=str(tw_frac)
    )


def load_instance(pth: str) -> RPInstance:
    """Load an instance in TSPLIB format,
    normalize it and wrap into RPInstance."""
    instance = read_tsplib_cvrptw(pth)
    instance = normalize_instance(instance)
    return instance


def dist_fit(x, plot: bool = True, kde_bdw_fac: float = 1.0):

    x_plt = np.linspace(0, x.max(), 100)

    # GAMMA
    gamma_param = stats.gamma.fit(x)
    gamma_fitted = stats.gamma(*gamma_param)
    #print(gamma_param)
    st, gamma_p_val= stats.ks_1samp(x, gamma_fitted.cdf)
    print(f"gamma p_val KS-test: {gamma_p_val}")
    k = len(gamma_param)
    gamma_ll = -gamma_fitted.logpdf(x).sum()
    gamma_aic = 2*k - 2*(gamma_ll)  # AIC - Akaike Information Criteria
    print(f"AIC: {gamma_aic}")
    if plot:
        ax = seaborn.histplot(x, stat="density")
        ax.plot(x_plt, gamma_fitted.pdf(x_plt), 'r-', lw=5, alpha=0.6)
        plt.show()

    # NORMAL
    normal_param = stats.norm.fit(x)
    normal_fitted = stats.norm(*normal_param)
    #print(normal_param)
    st2, normal_p_val = stats.ks_1samp(x, normal_fitted.cdf)
    print(f"normal p_val KS-test: {normal_p_val}")
    k = len(normal_param)
    normal_ll = -normal_fitted.logpdf(x).sum()
    normal_aic = 2*k - 2*(normal_ll)
    print(f"AIC: {normal_aic}")
    if plot:
        ax = seaborn.histplot(x, stat="density")
        ax.plot(x_plt, normal_fitted.pdf(x_plt), 'r-', lw=5, alpha=0.6)
        plt.show()

    # we have a bias towards GAMMA, since it simplifies sampling without need for
    # truncation in case of x < 0
    if gamma_p_val > 0.2:
        return {
            'dist': 'gamma',
            'params': gamma_param,
            'AIC': gamma_aic,
            'ks_test_p_val': gamma_p_val
        }
    elif (gamma_p_val < 0.1 and normal_p_val > 0.3 and normal_aic < gamma_aic) or \
            (gamma_p_val < 0.2 and normal_p_val > 0.3 and normal_aic < gamma_aic/2):
        return {
            'dist': 'normal',
            'params': normal_param,
            'AIC': normal_aic,
            'ks_test_p_val': normal_p_val
        }
    else:   # if nothing helps, use gaussian kernel density estimator
        # GAUSSIAN KDE
        if kde_bdw_fac < 1:
            bdw = lambda obj: np.power(obj.n, -1./(obj.d+4)) * kde_bdw_fac
            kde = stats.gaussian_kde(x, bw_method=bdw)
        else:
            kde = stats.gaussian_kde(x)
        cdf = lambda ary: np.array([kde.integrate_box_1d(-np.inf, p) for p in ary])
        _, kde_p_val = stats.kstest(x, cdf)
        print(f"kde p_val KS-test: {kde_p_val}")
        k = 1
        kde_ll = -kde.logpdf(x).sum()
        kde_aic = 2*k - 2*(kde_ll)
        print(f"AIC: {kde_aic}")
        if plot:
            ax = seaborn.histplot(x, stat="density")
            ax.plot(x_plt, kde.pdf(x_plt), 'r-', lw=5, alpha=0.6)
            plt.show()

        return {
            'dist': 'KDE',
            'params': kde,
            'AIC': kde_aic,
            'ks_test_p_val': kde_p_val
        }


def dist_fit_discrete(x, plot: bool = True):

    # POISSON
    # we need to map the discrete input (data) scale (e.g. 10, 20, 30, ...)
    # the scale of the poisson distribution (0, 1, 2, ...)
    unq = np.unique(x)
    num_unq = len(unq)
    unq_map = {u: i for u, i in zip(unq, np.arange(num_unq))}
    unq_map_inv = {i: u for u, i in zip(unq, np.arange(num_unq))}
    x_max = x.max() + np.std(x)/2

    x_mapped = np.array([unq_map[e] for e in x])
    #print(x_mapped)

    # maximum-likelihood estimator for the parameter of the
    # poisson distribution is the arithmetic sample mean
    poisson_param = [np.mean(x_mapped)]

    poisson_fitted = stats.poisson(*poisson_param)
    #print(poisson_param)
    st, poisson_p_val = stats.ks_1samp(x_mapped, poisson_fitted.cdf)
    print(f"poisson p_val KS-test: {poisson_p_val}")
    k = len(poisson_param)
    poisson_ll = -poisson_fitted.logpmf(x_mapped).sum()
    poisson_aic = 2*k - 2*(poisson_ll)  # AIC - Akaike Information Criteria
    print(f"AIC: {poisson_aic}")
    if plot:
        x_plt = np.arange(0, num_unq+2)
        ax = seaborn.histplot(x_mapped, stat="probability")
        ax.plot(x_plt, poisson_fitted.pmf(x_plt), 'r-', lw=5, alpha=0.6)
        plt.show()

        def test_function(x):
            x_ = np.array([unq_map[e] for e in x])
            return poisson_fitted.pmf(x_)

        ax = seaborn.histplot(x, stat="probability")
        ax.plot(unq, test_function(unq), 'r-', lw=5, alpha=0.6)
        plt.show()

        print(f"sample:")
        s = poisson_fitted.rvs(size=1000, random_state=np.random.default_rng(1))
        mode = int(list(unq_map_inv.values())[0])
        s = np.array([unq_map_inv[e] if e in list(unq_map_inv.keys()) else mode for e in s])
        seaborn.histplot(s, stat="count")
        plt.show()

    # GAMMA
    gamma_param = stats.gamma.fit(x)
    gamma_fitted = stats.gamma(*gamma_param)
    #print(gamma_param)
    st, gamma_p_val = stats.ks_1samp(x, gamma_fitted.cdf)
    print(f"gamma p_val KS-test: {gamma_p_val}")
    k = len(gamma_param)
    gamma_ll = -gamma_fitted.logpdf(x).sum()
    gamma_aic = 2*k - 2*(gamma_ll)  # AIC - Akaike Information Criteria
    print(f"AIC: {gamma_aic}")
    if plot:
        #x_plt = np.arange(0, num_unq+2)
        x_plt = np.linspace(0, x.max(), 100)
        ax = seaborn.histplot(x, stat="density")
        ax.plot(x_plt, gamma_fitted.pdf(x_plt), 'r-', lw=5, alpha=0.6)
        plt.show()

        print(f"sample:")
        # sample from gamma dist
        s = gamma_fitted.rvs(size=1000, random_state=np.random.default_rng(1))
        # round to bins
        max_diff = np.max(unq[1:]-unq[:-1])
        if max_diff < 1 + np.finfo(float).eps:
            s = np.round(s)     # integer range bins
        else:
            bin_centers = np.array([(l+u)/2 for l, u in zip(unq[:-1], unq[1:])])
            s = np.digitize(s, bin_centers)
            s = unq[s]

        # truncate if sample value is larger than half a std
        # from the max observed in the training data
        # and set to median value
        m = gamma_fitted.median()
        s[s > x_max] = m
        # plot...
        seaborn.histplot(s, stat="count")
        plt.show()

    # compare AIC to select better sampling model
    if (
            gamma_aic < poisson_aic or
            (gamma_p_val > 0.4 and poisson_p_val < 0.01) or
            (poisson_p_val < 1e-5 and gamma_p_val > poisson_p_val)
    ):
        if gamma_p_val < 0.1:
            warn(f"KS-test p_val of GAMMA is very small: {gamma_p_val}")
        return {
            'dist': 'gamma',
            'params': (gamma_param, (unq, x_max)),
            'AIC': gamma_aic,
            'ks_test_p_val': gamma_p_val
        }
    else:
        return {
            'dist': 'poisson',
            'params': (poisson_param, unq_map_inv),
            'AIC': poisson_aic,
            'ks_test_p_val': poisson_p_val
        }


def estimate_k(coords: np.ndarray, k_max: int = 15):
    """Use jump method to find most probable number of cluster centers."""

    distortion = []
    k_range = list(range(3, k_max+1))
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(coords)
        distortion.append(kmeans.inertia_**(-1))      # y = p/2 = 2/2 = 1

    jumps = [distortion[i]-distortion[i-1] for i in range(1, len(distortion))]
    max_jump = np.argmax(np.array(jumps))
    return k_range[max_jump], distortion[max_jump]


def analyze(instance: dict, buffer: dict = {}, plot: bool = True):

    org_df = instance['features'].loc[1:, :]
    tw_start_nz = org_df.tw_start[org_df.tw_start != 0]
    tw_end_nz = org_df.tw_end[org_df.tw_start != 0]
    tw_len = tw_end_nz-tw_start_nz

    id = instance['id']
    stat_summary = {
        'id': id,
        'org_tw_len': tw_len.mean(),
        'max_vehicle_number': instance['max_vehicle_number'],
        'vehicle_capacity': instance['vehicle_capacity'],
        #'dist_to_time_factor': instance['dist_to_time_factor'],
        'org_service_horizon': instance['org_service_horizon'],
        'summary': org_df.describe(),
    }

    df_w_depot = instance['norm_features']
    df = df_w_depot.loc[1:, :]
    stat_summary['norm_summary'] = df.describe()
    # points
    k, sig = 0, 0
    if "C" in id.upper():   # if clustered data
        k_max = 7 if "R" in id.upper() else 12
        k, sig = estimate_k(df_w_depot.loc[:, ('x_coord', 'y_coord')], k_max=k_max)
        print(f"num_clusters: {k}")
    stat_summary['n_components'] = k
    stat_summary['intra_cluster_var'] = sig
    if plot:
        ax = seaborn.scatterplot(x=df_w_depot['x_coord'], y=df_w_depot['y_coord'])
        ax.set_title(id)
        plt.show()

    # demand
    if plot:
        ax = seaborn.histplot(df.demand)
        ax.set_title(id)
        plt.show()
    stat_summary['demand'] = dist_fit_discrete(org_df.demand, plot=plot)    # use un-normalized integer demand!

    # tw start time
    tw_start_nz = df.tw_start[df.tw_start != 0]
    if plot:
        ax = seaborn.histplot(tw_start_nz)
        ax.set_title(id)
        ax.set_xlabel('TW start (no zero)')
        plt.show()
    stat_summary['tw_start'] = dist_fit(tw_start_nz, plot=plot)

    # tw size
    tw_end_nz = df.tw_end[df.tw_start != 0]
    tw_len = np.maximum(tw_end_nz-tw_start_nz, np.finfo(float).eps)
    tw_frac = len(tw_len) / df.shape[0]
    print(f"points with TW: {tw_frac}")
    print(f"TW len mean: {tw_len.mean()}")
    twl = tw_len.to_numpy()
    if np.allclose(twl, twl[0]):
        # tw len is constant
        stat_summary['tw_len'] = {
            'dist': 'const',
            'params': twl[0],
            'AIC': 0.0,
            'ks_test_p_val': 1.0
        }
    else:
        if plot:
            ax = seaborn.histplot(tw_len)
            ax.set_title(id)
            ax.set_xlabel('TW length (no zero/no limit)')
            plt.show()
        stat_summary['tw_len'] = dist_fit(tw_len, plot=plot)

    stat_summary['tw_frac'] = tw_frac

    buffer[f"tw_frac={tw_frac}"].append(stat_summary)
    return buffer


def process_group(path: str, types: list, plot: bool = True):

    out = {}
    for type in types:
        print(f"processing type: {type}")
        load_pth = os.path.join(path, type)
        file_names = os.listdir(load_pth)
        file_names.sort()

        data = []
        for fname in file_names:
            pth = os.path.join(load_pth, fname)
            print(f"preparing file '{fname}' from {pth}")
            instance = read_tsplib_cvrptw(pth)
            instance = normalize_instance(instance)
            data.append(instance)

        buffer = {'tw_frac=0.25': [], 'tw_frac=0.5': [], 'tw_frac=0.75': [], 'tw_frac=1.0': []}
        for inst in data:
            print(f"processing instance '{inst['id']}'")
            buffer = analyze(inst, buffer, plot=plot)

        for k, v in buffer.items():
            print(f"processed {len(v)} instances with {k}")

        out[type] = buffer

    return out

