import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import scanpy as sc
import anndata as ad
import cfp
from cfp.metrics import compute_metrics, compute_scalar_mmd
from ott.problems.linear import barycenter_problem
from ott.solvers.linear import sinkhorn, continuous_barycenter
from tqdm import tqdm
import sys
import os
from functools import partial, reduce
import gc

sys.path.append("/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/")
from utils import *

OUT_DIR = "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/results/baselines/"


def barycenter_interpolation(*args, epsilon=0.1, **kwargs):
    joint_data = np.concatenate(args, axis=0)
    num_per_segment = [data.shape[0] for data in args]
    bar_p = barycenter_problem.FreeBarycenterProblem(
        y=joint_data,
        epsilon=epsilon,
        num_per_segment=num_per_segment,
    )
    linear_solver = sinkhorn.Sinkhorn(lse_mode=True, **kwargs)
    solver = continuous_barycenter.FreeWassersteinBarycenter(linear_solver)
    out = solver(bar_p, bar_size=joint_data.shape[0])
    return out.x


def get_pred_barycenter(
    adata_train,
    cond,
):
    if len(cond.split("+")) == 2:
        single1, single2 = cond.split("+")
        pred_mean = barycenter_interpolation(
            adata_train.obsm["X_pca"][adata_train.obs["condition"] == single1],
            adata_train.obsm["X_pca"][adata_train.obs["condition"] == single2],
        )
    else:
        single1, single2, single3 = cond.split("+")
        pred_mean = barycenter_interpolation(
            adata_train.obsm["X_pca"][adata_train.obs["condition"] == single1],
            adata_train.obsm["X_pca"][adata_train.obs["condition"] == single2],
            adata_train.obsm["X_pca"][adata_train.obs["condition"] == single3],
        )
    return pred_mean


def get_pred_union(
    adata_train,
    cond,
):
    if len(cond.split("+")) == 2:
        single1, single2 = cond.split("+")
        singles = [single1, single2]
    else:
        single1, single2, single3 = cond.split("+")
        singles = [single1, single2, single3]
    pcas = [
        adata_train.obsm["X_pca"][adata_train.obs["condition"] == single]
        for single in singles
    ]
    n_cells = min([pca.shape[0] for pca in pcas])
    np.random.seed(42)
    pcas = [pca[np.random.choice(pca.shape[0], n_cells, replace=False)] for pca in pcas]
    pred_union = np.concatenate(pcas, axis=0)
    return pred_union


def get_pred_single(
    adata,
    adata_train,
    cond,
):
    if len(cond.split("+")) == 2:
        single1, single2 = cond.split("+")
        singles = [single1, single2]
    else:
        single1, single2, single3 = cond.split("+")
        singles = [single1, single2, single3]

    pcas = [adata.obsm["X_pca"][adata.obs["condition"] == single] for single in singles]
    pcas_train = [
        adata_train.obsm["X_pca"][adata_train.obs["condition"] == single]
        for single in singles
    ]
    mmds = [
        compute_scalar_mmd(
            adata.obsm["X_pca"][adata.obs["condition"] == cond],
            pca,
        )
        for pca in pcas
    ]
    pred_single = pcas_train[np.argmin(mmds)]
    return pred_single


def get_pred_closest(
    adata,
    adata_train,
    cond,
):
    pcas = [
        adata.obsm["X_pca"][adata.obs["condition"] == c]
        for c in adata_train.obs["condition"].unique()
    ]
    pcas_train = [
        adata_train.obsm["X_pca"][adata_train.obs["condition"] == c]
        for c in adata_train.obs["condition"].unique()
    ]
    mmds = [
        compute_scalar_mmd(
            adata.obsm["X_pca"][adata.obs["condition"] == cond],
            pca,
        )
        for pca in pcas
    ]
    pred_closest = pcas_train[np.argmin(mmds)]
    return pred_closest


def compute_baseline_metrics(
    adata,
    adata_train,
    adata_test,
    test_conds,
    method: Literal["barycenter", "union", "single", "closest"],
):
    adatas_pred = []
    for cond in tqdm(test_conds):
        # obtain prediction
        if method == "barycenter":
            pred = get_pred_barycenter(adata_train, cond)
        elif method == "union":
            pred = get_pred_union(adata_train, cond)
        elif method == "single":
            pred = get_pred_single(adata, adata_train, cond)
        elif method == "closest":
            pred = get_pred_closest(adata, adata_train, cond)
        # construct AnnData object
        adata_pred = ad.AnnData(X=np.zeros((pred.shape[0], adata.n_vars)))
        adata_pred.obs["dataset"] = "glut_post"
        adata_pred.obs["comb"] = adata_test.obs["comb"][
            adata_test.obs["condition"] == cond
        ].values[0]
        adata_pred.obs["condition"] = cond
        adata_pred.obsm["X_pca"] = pred
        adatas_pred.append(adata_pred)
    adata_pred = ad.concat(adatas_pred, join="outer")

    # reconstruct and reproject
    cfp.pp.reconstruct_pca(adata_pred, use_rep="X_pca", ref_adata=adata_train)
    adata_pred.X = csr_matrix(adata_pred.layers["X_recon"])
    del adata_pred.layers["X_recon"]
    cfp.pp.project_pca(adata_pred, adata, obsm_key_added="X_pca_reproj")

    # compute CellFlow metrics
    metrics_dfs = [
        compute_cfp_metrics(adata_eval, adata_pred, "X_pca", "X_pca", fast=False),
        compute_cfp_metrics(
            adata_eval, adata_pred, "X", "X", pred_name="recon", fast=False
        ),
        compute_cfp_metrics(
            adata_eval,
            adata_pred,
            "X_pca_all",
            "X_pca_reproj",
            pred_name="pca_reproj",
            fast=False,
        ),
    ]
    metrics_cfp = reduce(
        lambda left, right: pd.merge(left, right, on="condition"), metrics_dfs
    )

    # compute cluster-based metrics
    rep_gt = "X_pca"
    rep_pred = "X_pca_reproj"
    metrics_cluster = compute_cluster_metrics(
        adata,
        adata_pred,
        rep_gt,
        rep_pred,
        n_neighbors=[1, 10, 30],
        label_keys=["leiden_4", "final_clustering"],
    )

    # project predicted data on UMAP
    umap_fit_transform(adata, adata_pred, "X_pca", "X_pca_reproj")

    return adata_pred, metrics_cfp, metrics_cluster


if __name__ == "__main__":
    h5ad = "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons_glutpost.h5ad"
    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    test_combs_all = [
        ["RA+CHIR", "RA+CHIR+SHH", "RA+CHIR+BMP4"],
        ["FGF8+CHIR", "FGF8+CHIR+SHH", "FGF8+CHIR+BMP4"],
        # These ones we got already
        ["XAV+BMP4"],
        ["CHIR+BMP4", "RA+CHIR+BMP4", "FGF8+CHIR+BMP4"],
        ["RA+BMP4", "RA+CHIR+BMP4"],
        ["FGF8+BMP4", "FGF8+CHIR+BMP4"],
        ["XAV+SHH"],
        ["CHIR+SHH", "RA+CHIR+SHH", "FGF8+CHIR+SHH"],
        ["RA+SHH", "RA+CHIR+SHH"],
        ["FGF8+SHH", "FGF8+CHIR+SHH"],
        ["RA+CHIR+SHH"],
        ["RA+CHIR+BMP4"],
        ["FGF8+CHIR+SHH"],
        ["FGF8+CHIR+BMP4"],
    ]
    eval_combs_all = [
        ["RA+CHIR"],
        ["FGF8+CHIR"],
        # These ones we got already
        ["XAV+BMP4"],
        ["CHIR+BMP4"],
        ["RA+BMP4"],
        ["FGF8+BMP4"],
        ["XAV+SHH"],
        ["CHIR+SHH"],
        ["RA+SHH"],
        ["FGF8+SHH"],
        ["RA+CHIR+SHH"],
        ["RA+CHIR+BMP4"],
        ["FGF8+CHIR+SHH"],
        ["FGF8+CHIR+BMP4"],
    ]

    adatas_pred = {"barycenter": [], "union": [], "single": [], "closest": []}
    metrics_cfp_all = []
    metrics_cluster_all = []
    for test_combs, eval_combs in zip(test_combs_all, eval_combs_all):

        # load data
        molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
        adata = sc.read_h5ad(h5ad)

        # split data
        adata_train, adata_eval, sweep_conds = split_data(
            adata,
            test_dataset="glut_post",
            train_combs=[],
            exclude_combs=test_combs,
            eval_combs=eval_combs,
            sweep_combs=[],
            train_conds=[],
            eval_conds=[],
            sweep_conds=[],
        )

        # run PCA
        run_pca(adata, adata_train, adata_eval, "pca", n_dims_train=30, n_dims_eval=20)

        # compute and evaluate baselines
        test_conds = adata_eval.obs["condition"].unique()
        for method in ["barycenter", "union", "single", "closest"]:
            adata_pred, metrics_cfp, metrics_cluster = compute_baseline_metrics(
                adata,
                adata_train,
                adata_eval,
                test_conds,
                method=method,
            )
            metrics_cfp["method"] = method
            metrics_cluster["method"] = method
            adatas_pred[method].append(adata_pred)
            metrics_cfp_all.append(metrics_cfp)
            metrics_cluster_all.append(metrics_cluster)

        gc.collect()

    adatas_pred = {k: ad.concat(v, join="outer") for k, v in adatas_pred.items()}
    metrics_cfp_all = pd.concat(metrics_cfp_all, join="outer")
    metrics_cluster_all = pd.concat(metrics_cluster_all, join="outer")

    # save results
    for method, adata_pred in adatas_pred.items():
        adata_pred.write(os.path.join(OUT_DIR, f"adata_{method}.h5ad"))
    metrics_cfp_all.to_csv(
        os.path.join(OUT_DIR, "metrics_cfp_all.csv"),
        index=False,
    )
    metrics_cluster_all.to_csv(
        os.path.join(OUT_DIR, "metrics_cluster_all.csv"),
        index=False,
    )
