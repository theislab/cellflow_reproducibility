import scanpy as sc
import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple
import cfp
import scanpy as sc
import numpy as np
import functools
from ott.solvers import utils as solver_utils
import optax
from omegaconf import OmegaConf
from typing import NamedTuple, Any
import hydra
import wandb
import anndata as ad
import pandas as pd
import rapids_singlecell as rsc
import anndata as ad
import os
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca
from cfp.metrics._metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div


def compute_metrics(adata_ref: ad.AnnData, adata_ref_for_ct_error: ad.AnnData, adata_pred: ad.AnnData, adata_ood_true: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_broad", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    ct_true = adata_ood_true.obs[cell_type_col].value_counts().to_frame()
    ct_true = ct_true / ct_true.sum()
    
    compute_wknn(ref_adata=adata_ref_for_ct_error, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref_for_ct_error, label_key=cell_type_col)
    ct_transferred_pred = adata_pred.obs[f"{cell_type_col}_transfer"].value_counts().to_frame()
    ct_transferred_pred/=ct_transferred_pred.sum()
    all_cell_types = list(set(ct_true.index).union(set(ct_transferred_pred.index)))
    df_all_cell_types = pd.DataFrame(index=all_cell_types, data=np.zeros((len(all_cell_types), 2)), columns=["true", "pred"])
    df_all_cell_types["true"] = ct_true
    df_all_cell_types["pred"] = ct_transferred_pred
    df_all_cell_types = df_all_cell_types.fillna(0.0)
    cell_type_fraction_error = np.abs(df_all_cell_types["true"] - df_all_cell_types["pred"]).sum()
    
    all_cell_types_true = list(adata_ood_true.obs[cell_type_col].value_counts()[adata_ood_true.obs[cell_type_col].value_counts()>min_cells_for_dist_metrics].index)
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    
    e_distance = {}
    r_sq = {}
    mmd = {}
    sdiv_10 = {}
    sdiv_100 = {}
    n_cell_types_covered = 0
    for cell_type in all_cell_types_true: 
        dist_true = adata_ood_true[adata_ood_true.obs[f"{cell_type_col}"]==cell_type].obsm["X_aligned"]
        dist_pred = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type].obsm["X_aligned"]
        if len(dist_pred) == 0:
            continue
        n_cell_types_covered+=1
        r_sq[f"r_squared_{cell_type}"] = compute_r_squared(dist_true, dist_pred)
        e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
        mmd[f"mmd_{cell_type}"] = compute_scalar_mmd(dist_true, dist_pred)
        sdiv_10[f"div_10_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=10.0)
        sdiv_100[f"div_100_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=100.0)

    fraction_cell_types_covered = n_cell_types_covered/len(all_cell_types_true)


    # standard metrics
    ood_r_squared = compute_r_squared(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_sdiv_10 = compute_sinkhorn_div(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"], epsilon=10.0)
    ood_sdiv_100 = compute_sinkhorn_div(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"], epsilon=100.0)
    
    # metrics to return
    dict_to_log["fraction_cell_types_covered"] = fraction_cell_types_covered
    dict_to_log["cell_type_fraction_error"] = cell_type_fraction_error
    dict_to_log["mean_r_sq_per_cell_type"] = np.mean(list(r_sq.values()))
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(list(e_distance.values()))
    dict_to_log["mean_mmd_per_cell_type"] = np.mean(list(mmd.values()))
    dict_to_log["mean_sdiv_10_per_cell_type"] = np.mean(list(sdiv_10.values()))
    dict_to_log["mean_sdiv_100_per_cell_type"] = np.mean(list(sdiv_100.values()))
    dict_to_log["median_r_sq_per_cell_type"] = np.median(list(r_sq.values()))
    dict_to_log["median_e_distance_per_cell_type"] = np.median(list(e_distance.values()))
    dict_to_log["median_mmd_per_cell_type"] = np.median(list(mmd.values()))
    dict_to_log["median_sdiv_10_per_cell_type"] = np.median(list(sdiv_10.values()))
    dict_to_log["median_sdiv_100_per_cell_type"] = np.median(list(sdiv_100.values()))
    dict_to_log.update(r_sq)
    dict_to_log.update(e_distance)
    dict_to_log.update(mmd)
    dict_to_log.update(sdiv_10)
    dict_to_log.update(sdiv_100)
    dict_to_log["ood_r_squared"] = ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    dict_to_log["ood_sdiv_10"] = ood_sdiv_10
    dict_to_log["ood_sdiv_100"] = ood_sdiv_100
    return dict_to_log



if __name__ == "__main__":

    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/identity/zebrafish/single_condition_displacement_vector_same_perturbation"
    ood_conds = adata[adata.obs["gene_target"]!="control_control"].obs["condition"].unique()
    ood_cond_results = {}


    for ood_cond in ood_conds:
        print(ood_cond)
        adata_ood_true = adata[adata.obs["condition"]==ood_cond]
        adata_ref_for_ct_error = adata[adata.obs["condition"]!=ood_cond]
        tp = int(ood_cond.split("_")[-1])
        gene_knockout = "_".join(ood_cond.split("_")[:-1])
        adata_perturbed_other_timepoints = adata[(adata.obs["gene_target"]==gene_knockout) & (adata.obs["timepoint"]!=tp)]
        adata_control_other_timepoints = adata[(adata.obs["gene_target"]=="control_control") & (adata.obs["timepoint"]!=tp)]
        displacement_vecs = []
        for tp in adata_perturbed_other_timepoints.obs["timepoint"].unique():
            perturbed_mean = adata_perturbed_other_timepoints[adata_perturbed_other_timepoints.obs["timepoint"]==tp].obsm["X_aligned"].mean(axis=0)
            control_mean = adata_control_other_timepoints[adata_control_other_timepoints.obs["timepoint"]==tp].obsm["X_aligned"].mean(axis=0)
            displacement_vecs.append(perturbed_mean-control_mean)
        displacement_vector = np.array(displacement_vecs).mean(axis=0)
        adata_ood_pred = adata[(adata.obs["is_control"]) & (adata.obs["timepoint"]==tp)]
        adata_ood_pred.obsm["X_aligned"] = adata_ood_pred.obsm["X_aligned"] + displacement_vector
        
        if adata_ood_pred.n_obs > 30000:
            sc.pp.subsample(adata_ood_pred, n_obs=30000)
        if adata_ood_true.n_obs > 30000:
            sc.pp.subsample(adata_ood_true, n_obs=30000)

        ood_cond_results[ood_cond] = compute_metrics(adata_ref=adata, adata_ref_for_ct_error=adata_ref_for_ct_error, adata_pred=adata_ood_pred, adata_ood_true=adata_ood_true)

        pd.DataFrame.from_dict(ood_cond_results[ood_cond], columns=[ood_cond], orient="index").to_csv(os.path.join(out_dir, f"{ood_cond}_displacement_same_perturbation.csv"))

