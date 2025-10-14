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


def compute_metrics(adata_ref_for_ct_error: ad.AnnData, adata_pred: ad.AnnData, adata_ood_true: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_broad") -> dict:
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
    
    # metrics to return
    dict_to_log["cell_type_fraction_error"] = cell_type_fraction_error
    return dict_to_log


if __name__ == "__main__":

    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/identity/zebrafish/single_condition_min_cell_type_error"
    ood_conds = adata[adata.obs["gene_target"]!="control_control"].obs["condition"].unique()
    ood_cond_results = {}

    for ood_cond in ood_conds:
        print(ood_cond)
        adata_ood_true = adata[adata.obs["condition"]==ood_cond]
        adata_ref_for_ct_error = adata[adata.obs["condition"]!=ood_cond]
        tp = int(ood_cond.split("_")[-1])

        
        if adata_ood_true.n_obs > 30000:
            sc.pp.subsample(adata_ood_true, n_obs=30000)
        ood_cond_results[ood_cond] = compute_metrics(adata_ref_for_ct_error=adata_ref_for_ct_error, adata_pred=adata_ood_true, adata_ood_true=adata_ood_true)

        pd.DataFrame.from_dict(ood_cond_results[ood_cond], columns=[ood_cond], orient="index").to_csv(os.path.join(out_dir, f"{ood_cond}_min_ct_error.csv"))

