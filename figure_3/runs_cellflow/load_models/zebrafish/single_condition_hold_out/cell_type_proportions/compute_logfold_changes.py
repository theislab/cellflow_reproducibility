import pandas as pd
import numpy as np
import scanpy as sc
import os
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca
import sys
import anndata as ad

ood_cond, wandb_name = sys.argv[1], sys.argv[2]

tp = int(ood_cond.split("_")[-1])
data_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/single_condition_f"
out_dir = "/lustre/groups/ml01/workspace/cell_flow_zebrafish/results/single_condition/cell_type_proportion"


pred_file = f"{wandb_name}_{ood_cond}_pred.h5ad"
adata_preds = sc.read_h5ad(os.path.join(data_dir, pred_file))

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")

compute_wknn(ref_adata=adata, query_adata=adata_preds, n_neighbors=1, ref_rep_key="X_aligned", query_rep_key="X_aligned")
transfer_labels(query_adata=adata_preds, ref_adata=adata, label_key="cell_type_broad")
adata_preds.obs["cell_type_broad"] = adata_preds.obs["cell_type_broad_transfer"]

adata_true = adata[adata.obs["condition"]==ood_cond]
adata_ctrl = adata[(adata.obs["timepoint"]==tp) & (adata.obs["gene_target"]=="control_control")]


ct_pred = adata_preds.obs[f"cell_type_broad_transfer"].value_counts().to_frame()
ct_pred/=ct_pred.sum()

ct_true = adata_true.obs[f"cell_type_broad"].value_counts().to_frame()
ct_true/=ct_true.sum()

ct_ctrl = adata_ctrl.obs[f"cell_type_broad"].value_counts().to_frame()
ct_ctrl/=ct_ctrl.sum()

ct_pred["ctrl"] = ct_ctrl["count"]
ct_true["ctrl"] = ct_ctrl["count"]
ct_pred["ratio_pred"] = ct_pred["count"]/ct_pred["ctrl"]
ct_true["ratio_true"] = ct_true["count"]/ct_true["ctrl"]
ct_pred["ratio_true"] = ct_true["ratio_true"]
ct_pred = ct_pred[["ratio_pred", "ratio_true"]]
ct_pred["logratio_true"] = np.log2(ct_pred["ratio_true"])
ct_pred["logratio_pred"] = np.log2(ct_pred["ratio_pred"])

ct_pred.to_csv(os.path.join(out_dir, f"{ood_cond}_cell_type_proportions.csv"))


