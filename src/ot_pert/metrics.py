from typing import Dict, Any
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances

from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn
from ott.solvers import linear
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))

def compute_sinkhorn_div(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    return float(sinkhorn_divergence(
        pointcloud.PointCloud,
        x=x,
        y=y,
        cost_fn=costs.SqEuclidean(),
        epsilon=epsilon,
        scale_cost=1.0,
    ).divergence)

def compute_e_distance(x: np.ndarray, y: np.ndarray) -> float:
	delta = 1 / (len(x) * len(y)) * euclidean_distances(x, y).sum()
	sigma_1 = 1 / ((len(x)-1) * len(x)) * euclidean_distances(x, x).sum()
	sigma_2 = 1 / ((len(y)-1) * len(y)) * euclidean_distances(y, y).sum()

	return 2 * delta - sigma_1 - sigma_2

def compute_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_01"] = compute_sinkhorn_div(x, y, epsilon=0.1)
    #metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    #metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd"] = compute_sinkhorn_div(x, y, epsilon=100_000)
    return metrics


def compute_mean_metrics(metrics: Dict[str, Dict[str, float]], prefix: str = ""):
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict = {prefix+met_name: [] for met_name in metric_names}
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix+met] = stat/len(metrics)
    return metric_dict
