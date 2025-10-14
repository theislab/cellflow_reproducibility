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
import os
import pickle
from cfp.training import ComputationCallback
from cfp.preprocessing import transfer_labels, compute_wknn
from cfp.training import ComputationCallback
from numpy.typing import ArrayLike
from cfp.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
from cfp.metrics import compute_r_squared, compute_e_distance, compute_metrics_fast
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca




def compute_metrics(adata_ref: ad.AnnData, adata_pred: ad.AnnData, donor_deg_dict: dict, adata_ood_true: ad.AnnData, adata_ctrl: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_new", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_pca", query_rep_key="X_pca_for_ct_transfer")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    
    e_distance = {}
    r_sq = {}
    mmd = {}
    sdiv_10 = {}
    sdiv_100 = {}
    deg_e_distance = {}
    deg_r_sq = {}
    deg_mmd = {}
    deg_sdiv_10 = {}
    deg_sdiv_100 = {}
    for ct_cyto in donor_deg_dict.keys(): 
        cell_type = ct_cyto.split("_")[1]
        adata_true_ct = adata_ood_true[(adata_ood_true.obs[f"{cell_type_col}"]==cell_type)]
        adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type]
        if adata_pred_ct.n_obs == 0:
            continue
        dist_true_decoded = adata_true_ct.X.toarray()
        dist_pred_decoded = adata_pred_ct.X
        dist_true = adata_true_ct.obsm["X_pca"]
        dist_pred = adata_pred_ct.obsm["X_pca"]
        r_sq[f"decoded_r_squared_{cell_type}"] = compute_r_squared(dist_true_decoded, dist_pred_decoded)
        e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
        mmd[f"mmd_{cell_type}"] = compute_scalar_mmd(dist_true, dist_pred)
        sdiv_10[f"div_10_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=10.0)
        sdiv_100[f"div_100_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=100.0)

        deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
        deg_true_decoded = adata_true_ct[:,deg_mask].X.toarray()
        deg_pred_decoded = adata_pred_ct[:,deg_mask].X
        deg_r_sq[f"deg_decoded_r_squared_{cell_type}"] = compute_r_squared(deg_true_decoded, deg_pred_decoded)
        deg_e_distance[f"deg_e_distance_{cell_type}"] = compute_e_distance(deg_true_decoded, deg_pred_decoded)
        deg_mmd[f"deg_mmd_{cell_type}"] = compute_scalar_mmd(deg_true_decoded, deg_pred_decoded)
        deg_sdiv_10[f"deg_div_10_{cell_type}"] = compute_sinkhorn_div(deg_true_decoded, deg_pred_decoded, epsilon=10.0)
        deg_sdiv_100[f"deg_div_100_{cell_type}"] = compute_sinkhorn_div(deg_true_decoded, deg_pred_decoded, epsilon=100.0)

    adata_concat = ad.concat([adata_ctrl, adata_pred], join="inner", label="all")
    sc.tl.rank_genes_groups(
            adata_concat,
            groupby="cytokine",
            reference="PBS",
            rankby_abs=True,
            n_genes=50,
            use_raw=False,
            method="wilcoxon",
        )
    predicted_deg_genes = [el[0] for el in list(adata_concat.uns["rank_genes_groups"]["names"])]

    # standard metrics
    decoded_ood_r_squared = compute_r_squared(adata_ood_true.X.toarray(), adata_pred.X)
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_sdiv_10 = compute_sinkhorn_div(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"], epsilon=10.0)
    ood_sdiv_100 = compute_sinkhorn_div(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"], epsilon=100.0)
    
    # metrics to return
    dict_to_log["mean_decoded_r_sq_per_cell_type"] = np.mean(list(r_sq.values()))
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(list(e_distance.values()))
    dict_to_log["mean_mmd_per_cell_type"] = np.mean(list(mmd.values()))
    dict_to_log["mean_sdiv_10_per_cell_type"] = np.mean(list(sdiv_10.values()))
    dict_to_log["mean_sdiv_100_per_cell_type"] = np.mean(list(sdiv_100.values()))
    dict_to_log["mean_deg_r_sq_per_cell_type"] = np.mean(list(deg_r_sq.values()))
    dict_to_log["mean_deg_e_distance_per_cell_type"] = np.mean(list(deg_e_distance.values()))
    dict_to_log["mean_deg_mmd_per_cell_type"] = np.mean(list(deg_mmd.values()))
    dict_to_log["mean_deg_sdiv_10_per_cell_type"] = np.mean(list(deg_sdiv_10.values()))
    dict_to_log["mean_deg_sdiv_100_per_cell_type"] = np.mean(list(deg_sdiv_100.values()))
    
    dict_to_log.update(r_sq)
    dict_to_log.update(e_distance)
    dict_to_log.update(mmd)
    dict_to_log.update(sdiv_10)
    dict_to_log.update(sdiv_100)
    dict_to_log.update(deg_r_sq)
    dict_to_log.update(deg_e_distance)
    dict_to_log.update(deg_mmd)
    dict_to_log.update(deg_sdiv_10)
    dict_to_log.update(deg_sdiv_100)
    dict_to_log["decoded_ood_r_squared"] = decoded_ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    dict_to_log["ood_sdiv_10"] = ood_sdiv_10
    dict_to_log["ood_sdiv_100"] = ood_sdiv_100
    dict_to_log["predicted_deg_genes"] = predicted_deg_genes
    return dict_to_log



if __name__ == "__main__":
    pred_file = sys.argv[1]
    complete_pred_file = os.path.join("/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor", pred_file)
    adata_pred = sc.read_h5ad(complete_pred_file)
    adata_pred.X = adata_pred.layers["X_recon"]
    cytokine = pred_file.split("_")[-3]
    donor = pred_file.split("_")[-4]
    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor"
    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ood_true = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
    adata_ctrl = adata_full[(adata_full.obs["cytokine"]=="PBS") & (adata_full.obs["donor"]==donor)]
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)
    
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
    project_pca(query_adata=adata_pred, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")
    project_pca(query_adata=adata_pred, ref_adata=adata_full, obsm_key_added="X_pca")
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs.pkl", "rb") as pickle_file:
        deg_genes = pickle.load(pickle_file)
    donor_deg_dict = {k: v for k, v in deg_genes.items() if (k.startswith(donor) and k.endswith(f"_{cytokine}"))}
    adata_pred.obs["cytokine"] = cytokine
    adata_pred.obs["donor"] = donor
    adata_pred.var_names=adata_ctrl.var_names   
    out = compute_metrics(adata_ref=adata_ref, adata_pred=adata_pred, donor_deg_dict=donor_deg_dict, adata_ood_true=adata_ood_true, adata_ctrl=adata_ctrl)
    out["wandb_name"] = pred_file.split("_")[0]
    out["cytokine_in_train"] = adata_pred.uns["cytokine_in_train"]
    out["cytokine_in_train"] = len(adata_pred.uns["cytokine_in_train"])
    cond = f'{donor}_{cytokine}'
    expr = "_".join(pred_file.split("_")[:-1])
    pd.DataFrame.from_dict(out, columns=[cond], orient="index").to_csv(os.path.join(out_dir, f"{expr}_metrics.csv"))
    

