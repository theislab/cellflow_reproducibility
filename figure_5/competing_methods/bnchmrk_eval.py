import sys
import os
import warnings
import yaml
from typing import Literal
from functools import reduce
import numpy as np
import pandas as pd
import scanpy as sc
import inspect

sys.path.append("/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/")
from utils import *

warnings.filterwarnings("ignore")


def eval_otfm_ineurons(
    *,
    h5ad: str,
    train_datasets: list[Literal["glut_post", "glut_pre", "gaba_post", "gaba_pre"]],
    test_dataset: Literal["glut_post", "gaba_post"],
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
    plot_umap_combs: list[str],
    plot_umap_conds: list[str],
    plot_heatmap_combs: list[str],
    plot_heatmap_conds: list[str],
    n_dims_eval: int,
    out_dir: str,
    out_prefix: str,
    save_adata: bool,
    use_counts: bool = False,
    minimal: bool = True,
) -> None:

    ####################
    #### Load data #####
    ####################

    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    adata = sc.read_h5ad(h5ad)
    adata = adata[adata.obs["dataset"].isin(train_datasets)]
    adata_pred = sc.read_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")

    ####################
    ### Prepare data ###
    ####################

    # split data
    adata_train, adata_eval, sweep_conds = split_data(
        adata,
        test_dataset,
        train_combs,
        exclude_combs,
        eval_combs,
        sweep_combs,
        train_conds,
        eval_conds,
        sweep_conds,
    )

    # run PCA
    run_pca(
        adata,
        adata_train,
        adata_eval,
        latent_space="pca",
        n_dims_train=1,
        n_dims_eval=n_dims_eval,
    )

    # choose counts or log1p
    if use_counts:
        adata.X = adata.layers["counts"]

    ####################
    #### Evaluate OT ###
    ####################

    # reproject PCA
    cfp.pp.project_pca(adata_pred, adata, obsm_key_added="X_pca_reproj")

    # compute cfp metrics (e-distance, mmd, r2)
    if minimal:
        metrics_cfp = compute_cfp_metrics(
            adata_eval,
            adata_pred,
            "X_pca_all",
            "X_pca_reproj",
            pred_name="pca_reproj",
        )
    else:
        metrics_dfs = [
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
    if len(sweep_conds) > 0:
        metrics_cfp["sweep"] = metrics_cfp["condition"].isin(sweep_conds)

    # compute cluster metrics
    if not minimal:
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

        # save outputs
        metrics_cfp.to_csv(f"{out_dir}/{out_prefix}_metrics_cfp.csv", index=False)
        metrics_cluster.to_csv(
            f"{out_dir}/{out_prefix}_metrics_cluster.csv", index=False
        )
        if save_adata:
            adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")
        else:
            os.remove(f"{out_dir}/{out_prefix}_pred.h5ad")

        # visualize predictions on a UMAP
        figs_dir = os.path.join(out_dir, "figs")
        plot_predictions_umap(
            adata, adata_pred, "X_umap", plot_umap_combs, plot_umap_conds, figs_dir
        )

        # visualize predictions on a heatmap
        plot_predictions_heatmap(
            adata,
            adata_train,
            adata_pred,
            plot_combs=plot_heatmap_combs,
            plot_conds=plot_heatmap_conds,
            out_dir=figs_dir,
            label_keys=["leiden_4", "final_clustering"],
        )

    else:
        # save outputs
        metrics_cfp.to_csv(f"{out_dir}/{out_prefix}_metrics_cfp.csv", index=False)
        if save_adata:
            adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")
        else:
            os.remove(f"{out_dir}/{out_prefix}_pred.h5ad")


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs(os.path.join(config["out_dir"], "figs"), exist_ok=True)
    fn_sig = inspect.signature(eval_otfm_ineurons)
    config = {k: v for k, v in config.items() if k in fn_sig.parameters}
    eval_otfm_ineurons(**config)
