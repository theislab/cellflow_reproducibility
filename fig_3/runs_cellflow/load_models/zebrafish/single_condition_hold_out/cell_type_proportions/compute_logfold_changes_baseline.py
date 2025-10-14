import pandas as pd
import numpy as np
import scanpy as sc
import os
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca
import sys
import anndata as ad

ood_cond, wandb_name = sys.argv[1], sys.argv[2]

out_dir = "/lustre/groups/ml01/workspace/cell_flow_zebrafish/results/single_condition/cell_type_proportion_baseline"

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")


for tp in adata.obs["timepoint"].unique():
    adata_ctrl = adata[adata.obs["condition"]==f"control_control_{tp}"]

    for single_fish in adata_ctrl.obs["Oligo"].unique():
        adata_single = adata_ctrl[adata_ctrl.obs["Oligo"]==single_fish]
        adata_rest = adata_ctrl[adata_ctrl.obs["Oligo"]!=single_fish]

        ct_single_fish = adata_single.obs[f"cell_type_broad"].value_counts().to_frame()
        ct_single_fish/=ct_single_fish.sum()

        ct_rest = adata_rest.obs[f"cell_type_broad"].value_counts().to_frame()
        ct_rest/=ct_rest.sum()

        ct_single_fish["ratio"] = ct_single_fish["count"]/ct_rest["ctrl"]
        ct_single_fish["logratio"] = np.log2(ct_single_fish["ratio_true"])
        
        ct_single_fish.to_csv(os.path.join(out_dir, f"{ood_cond}_cell_type_proportions.csv"))


