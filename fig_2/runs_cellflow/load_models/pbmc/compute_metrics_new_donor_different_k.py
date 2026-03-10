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
    pred_file = sys.argv[1]
    complete_pred_file = os.path.join("/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor", pred_file)
    adata_pred = sc.read_h5ad(complete_pred_file)
    adata_pred.X = adata_pred.layers["X_recon"]
    cytokine = pred_file.split("_")[-3]
    donor = pred_file.split("_")[-4]
    donor_held_out = donor
    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_different_k"
    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ood_true = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
    adata_ctrl = adata_full[(adata_full.obs["cytokine"]=="PBS") & (adata_full.obs["donor"]==donor)]

    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)

    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
    project_pca(query_adata=adata_pred, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")

    
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs_different_top_k.pkl", "rb") as pickle_file:
        deg_genes = pickle.load(pickle_file)

    out = compute_metrics(adata_ref=adata_ref, adata_pred=adata_pred, deg_dict=deg_genes, adata_ood_true=adata_ood_true, adata_ctrl=adata_ctrl)
    df = pd.DataFrame.from_dict(out)
    condition = f"{donor_held_out}_{cytokine}"
    df["condition"]=condition
    df.to_csv(os.path.join(out_dir, f"{pred_file}_{condition}.csv"))

    

