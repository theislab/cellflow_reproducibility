import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple
import cellflow
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
from cellflow.training import ComputationCallback
from cellflow.preprocessing import transfer_labels, compute_wknn
from cellflow.training import ComputationCallback
from numpy.typing import ArrayLike
from cellflow.metrics import compute_r_squared, compute_e_distance
from cellflow.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
import sys
import pickle
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca



def compute_metrics(adata_ref: ad.AnnData, adata_pred: ad.AnnData, deg_dict: dict, adata_ood_true: ad.AnnData, adata_ctrl: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_new", min_cells_for_dist_metrics: int = 50) -> dict:
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_pca", query_rep_key="X_pca_for_ct_transfer")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)

    deg_r_sq = {}
    for k in deg_genes.keys():
        donor_deg_dict = {k: v for k, v in deg_genes[k].items() if (k.startswith(donor_held_out) and k.endswith(f"_{cytokine}"))}
        deg_r_sq[k] = {}
        for ct_cyto in donor_deg_dict.keys():
            cell_type = ct_cyto.split("_")[1]
            adata_true_ct = adata_ood_true[(adata_ood_true.obs[f"{cell_type_col}"]==cell_type)]
            adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type]
            if adata_pred_ct.n_obs == 0:
                continue
        
            deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
            deg_true_decoded = adata_true_ct[:,deg_mask].X.toarray()
            deg_pred_decoded = adata_pred_ct[:,deg_mask].X
            deg_r_sq[k][f"deg_decoded_r_squared_{cell_type}"] = compute_r_squared(deg_true_decoded, deg_pred_decoded)
        
      
    return deg_r_sq


if __name__ == "__main__":
    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/additive_model/pbmc_new_donor/identity_different_k"
    donor_held_out = sys.argv[1]
    idx_given_donor = sys.argv[2]

    control_key = "is_control"
        
    adata_train = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{donor_held_out}/{str(idx_given_donor)}/adata_train_{donor_held_out}.h5ad")
    adata_ood_perturbed  = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{donor_held_out}/{str(idx_given_donor)}/adata_ood_{donor_held_out}.h5ad")
    cytokines_to_impute = adata_train.uns["split_info"][idx_given_donor]["cytokines_to_impute"]
    cytokines_to_train_data = adata_train.uns["split_info"][idx_given_donor]["cytokines_to_train_data"]
    if len(cytokines_to_train_data) != 65:
        sys.exit(0)

    adata_ctrl = adata_train[adata_train.obs[control_key].to_numpy()]


    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)
    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
    
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs_different_top_k.pkl", "rb") as pickle_file:
        deg_genes = pickle.load(pickle_file)

    adata_ctrl_current_donor = adata_ctrl[adata_ctrl.obs["donor"]==donor_held_out]
    if adata_ctrl_current_donor.n_obs > 10000:
            sc.pp.subsample(adata_ctrl_current_donor, n_obs=10000)
    
    for cytokine in cytokines_to_impute:
        pred1 = adata_ctrl_current_donor.X.toarray()
        condition = f"{donor_held_out}_{cytokine}"
        conditions = [condition] * pred1.shape[0]

        obs_data = pd.DataFrame({
            'condition': conditions
        })

        adata_pred = ad.AnnData(X=pred1, obs=obs_data)
        adata_pred.obs["cytokine"] = cytokine
        adata_pred.obs["donor"] = donor_held_out
        adata_pred.var_names=adata_ctrl.var_names   
        
        project_pca(query_adata=adata_pred, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")
        project_pca(query_adata=adata_pred, ref_adata=adata_full, obsm_key_added="X_pca")
        cond_orig = condition
        condition = condition + "_" + str(len(cytokines_to_train_data))
        adata_ood_true = adata_full[(adata_full.obs["donor"] == donor_held_out) & (adata_full.obs["cytokine"]==cytokine)]
        
        out = compute_metrics(adata_ref=adata_ref, adata_pred=adata_pred, deg_dict=deg_genes, adata_ood_true=adata_ood_true, adata_ctrl=adata_ctrl_current_donor)
        df = pd.DataFrame.from_dict(out)
        df["condition"]=condition
        df["num_cytokines_in_train"] = len(cytokines_to_train_data)
        
        df.to_csv(os.path.join(out_dir, f"{idx_given_donor}_{condition}.csv"))
