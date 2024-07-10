from typing import Dict

import numpy as np
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel
import math


def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    return float(
        sinkhorn_divergence(
            pointcloud.PointCloud,
            x=x,
            y=y,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        ).divergence
    )


def compute_e_distance(x: np.ndarray, y: np.ndarray) -> float:
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics


def compute_mean_metrics(metrics: Dict[str, Dict[str, float]], prefix: str = ""):
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict = {prefix + met_name: [] for met_name in metric_names}
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):  # from CellOT repo
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def sample_rows(array, num_rows):
    n, m = array.shape
    if num_rows > n:
        raise ValueError("num_rows cannot be greater than the number of rows in the array.")
    
    # Randomly choose indices of rows to sample
    row_indices = np.random.choice(n, math.ceil(num_rows), replace=False)

    return row_indices

def compute_metrics_fast(x: np.ndarray, y: np.ndarray, subsample: bool = True) -> Dict[str, float]:
    if subsample:
        indices = sample_rows(x, x.shape[0]*0.1)
        x = x[indices, :]
        y = y[indices, :]
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd_distance"] = compute_scalar_mmd(x, y)
    return metrics
