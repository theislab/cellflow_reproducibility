import pandas as pd
import numpy as np
import scanpy as sc
import os
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca
import sys
import anndata as ad
import pandas as pd

ood_cond = sys.argv[1]

tp = int(ood_cond.split("_")[-1])
data_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/single_condition_f"
out_dir = "/lustre/groups/ml01/workspace/cell_flow_zebrafish/results/single_condition/cell_type_proportion_mean_model"


adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
adata_ref_for_ct_error = adata[adata.obs["condition"]!=ood_cond]
tp = int(ood_cond.split("_")[-1])
gene_knockout = "_".join(ood_cond.split("_")[:-1])
adata_perturbed_same_timepoint = adata[(adata.obs["gene_target"]!=gene_knockout) & (adata.obs["timepoint"]==tp) & (adata.obs["gene_target"]!="control_control")]
adata_ood_pred = adata[(adata.obs["is_control"]) & (adata.obs["timepoint"]==tp)]
control_mean = adata_ood_pred.obsm["X_aligned"].mean(axis=0)

displacement_vecs = []
for pert in adata_perturbed_same_timepoint.obs["gene_target"].unique():
    perturbed_mean = adata_perturbed_same_timepoint[adata_perturbed_same_timepoint.obs["gene_target"]==pert].obsm["X_aligned"].mean(axis=0)
    displacement_vecs.append(perturbed_mean-control_mean)
displacement_vector = np.array(displacement_vecs).mean(axis=0)
adata_ood_pred.obsm["X_aligned"] = adata_ood_pred.obsm["X_aligned"] + displacement_vector

if adata_ood_pred.n_obs > 30000:
    sc.pp.subsample(adata_ood_pred, n_obs=30000)

adata_preds = adata_ood_pred
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


