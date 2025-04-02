import jax 
jax.config.update("jax_enable_x64", True)
import cfp
import scanpy as sc
import anndata as ad
import numpy as np
import cfp.preprocessing as cfpp
import pandas as pd
import os
import sys
from cfp.metrics import compute_metrics_fast
from cfp.preprocessing import transfer_labels, compute_wknn
from cfp.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div

def compute_metrics(adata_ref: ad.AnnData, adata_pred: ad.AnnData, adata_ood_true: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_broad", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    compute_wknn(ref_adata=adata_ref, query_adata=adata_ood_true, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_ood_true, ref_adata=adata_ref, label_key=cell_type_col)
    ct_transferred_true = adata_ood_true.obs["cell_type_broad_transfer"].value_counts().to_frame()
    ct_transferred_true = ct_transferred_true / ct_transferred_true.sum()
    
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    ct_transferred_pred = adata_pred.obs[f"{cell_type_col}_transfer"].value_counts().to_frame()
    ct_transferred_pred/=ct_transferred_pred.sum()
    all_cell_types = list(set(ct_transferred_true.index).union(set(ct_transferred_pred.index)))
    df_all_cell_types = pd.DataFrame(index=all_cell_types, data=np.zeros((len(all_cell_types), 2)), columns=["true", "pred"])
    df_all_cell_types["true"] = ct_transferred_true
    df_all_cell_types["pred"] = ct_transferred_pred
    df_all_cell_types = df_all_cell_types.fillna(0.0)
    cell_type_fraction_error = np.abs(df_all_cell_types["true"] - df_all_cell_types["pred"]).sum()
    
    all_cell_types_true = list(adata_ood_true.obs["cell_type_broad"].value_counts()[adata_ood_true.obs["cell_type_broad"].value_counts()>min_cells_for_dist_metrics].index)

    e_distance = {}
    r_sq = {}
    mmd = {}
    n_cell_types_covered = 0
    for cell_type in all_cell_types_true: 
        dist_true = adata_ood_true[adata_ood_true.obs["cell_type_broad"]==cell_type].obsm["X_pca_from_aligned"]
        dist_pred = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type].obsm["X_pca_from_aligned"]
        if len(dist_pred) == 0:
            continue
        n_cell_types_covered+=1
        r_sq[f"r_squared_{cell_type}"] = compute_r_squared(dist_true, dist_pred)
        e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
        mmd[f"mmd_{cell_type}"] = compute_scalar_mmd(dist_true, dist_pred)
        
    fraction_cell_types_covered = n_cell_types_covered/len(all_cell_types_true)


    # standard metrics
    ood_r_squared = compute_r_squared(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    
    # metrics to return
    dict_to_log["fraction_cell_types_covered"] = fraction_cell_types_covered
    dict_to_log["cell_type_fraction_error"] = cell_type_fraction_error
    dict_to_log["mean_r_sq_per_cell_type"] = np.mean(list(r_sq.values()))
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(list(e_distance.values()))
    dict_to_log["mean_mmd_per_cell_type"] = np.mean(list(mmd.values()))
    dict_to_log["median_r_sq_per_cell_type"] = np.median(list(r_sq.values()))
    dict_to_log["median_e_distance_per_cell_type"] = np.median(list(e_distance.values()))
    dict_to_log["median_mmd_per_cell_type"] = np.median(list(mmd.values()))
    dict_to_log.update(r_sq)
    dict_to_log.update(e_distance)
    dict_to_log.update(mmd)
    dict_to_log["ood_r_squared"] = ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    return dict_to_log


if __name__ == "__main__":
    args = sys.argv
    ood_cond, model_name = args[1], args[2]

    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/identity/zebrafish/full_gene_hyperparameter_search"

    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    control_key = "is_control"
    adata_ood_true = adata[adata.obs["gene_target"]==ood_cond]
    adata_ood_perturbed = adata[adata.obs["gene_target"]==ood_cond]
    adata_ctrl = adata[adata.obs[control_key].to_numpy()]
    adata_ctrl_subsetted = []
    for tp in adata_ctrl.obs["timepoint"].unique():
        adata_tmp = adata_ctrl[adata_ctrl.obs["timepoint"]==tp]
        sc.pp.subsample(adata_tmp, n_obs=30000)
        adata_ctrl_subsetted.append(adata_tmp)
    adata_ctrl = ad.concat(adata_ctrl_subsetted)

    adata_ctrl.uns = adata.uns.copy()
    adata_ood_perturbed.uns = adata.uns.copy()



    cf = cfp.model.CellFlow.load(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex_final/{model_name}_CellFlow.pkl")

    covariate_data = adata_ood_perturbed.obs.drop_duplicates(subset=["condition"])
    preds = cf.predict(adata=adata_ctrl, sample_rep="X_aligned", condition_id_key="condition", covariate_data=covariate_data)


    cell_type_col = "cell_type_broad"

    for cond, array in preds.items():
        ref_adata = adata[adata.obs["condition"]!=cond]
        metrics = compute_metrics_fast(adata[adata.obs["condition"]==cond].obsm["X_aligned"], array)
        
        obs_data = pd.DataFrame({
            'condition': [cond] * array.shape[0]
        })
        adata_pred = ad.AnnData(X=np.empty((len(array),1)), obs=obs_data)
        adata_pred.obsm["X_aligned"] = array
        compute_wknn(ref_adata=ref_adata, query_adata=adata_pred, n_neighbors=1, ref_rep_key="X_aligned", query_rep_key="X_aligned")
        transfer_labels(query_adata=adata_pred, ref_adata=ref_adata, label_key=cell_type_col)
                
        out = compute_metrics(adata_ref=ref_adata, adata_pred=adata_pred, adata_ood_true=adata_ood_true)
        pd.DataFrame.from_dict(out, columns=[cond], orient="index").to_csv(os.path.join(out_dir, f"{model_name}_{cond}.csv"))

