import functools
import jax
import numpy as np
import scanpy as sc
import cfp.preprocessing as cfpp
from cfp.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
import os
import pandas as pd
import sys
from pathlib import Path

def mean_model(adata_pert, adata_ctrl):
    # Mean displacement of same donor across cytokines
    displacement = adata_pert.X.toarray().mean(axis=0) - adata_ctrl.X.toarray().mean(axis=0)
    pred = np.asarray(adata_ctrl.X.toarray() + displacement)
    return pred

print("Start")
split = sys.argv[1]
path_to_splits = Path(sys.argv[2])
task = sys.argv[3]
save_path = sys.argv[4]
ref_path = sys.argv[5]

# Read the reference dataset plus centered PCA 
adata_ref = sc.read_h5ad(ref_path)
print("Centered pca")
cfpp.centered_pca(adata_ref, n_comps=20, keep_centered_data=False)

# Read the real datasets 
adata_train_path = path_to_splits / f"adata_train_{split}.h5ad"
adata_ood_path = path_to_splits / f"adata_ood_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path) 
adata_ood = sc.read_h5ad(adata_ood_path)
print("Read ref")

# Projected PCA of the test datasets on the reference 
print("Projected pca")
cfpp.project_pca(query_adata=adata_train, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)

adata_train_vars = adata_train.var_names

# Collect observations 
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}

cond_name = "perturbation_condition" if task=="ood_genes" else "condition"
for cond in adata_ood.obs[cond_name].cat.categories:
    print(f"Condition {cond}")
    if "NT" in cond:
        continue
    # Extract pathway from conditions
    pathway = cond.split("_")[1]  # Actually always the same 
    # Take perturbed dataset for the given pathway 
    adata_pred_train = adata_train[adata_train.obs["pathway"] == pathway]
    adata_pred_train = adata_pred_train[~adata_pred_train.obs["control"]]
    adata_control_copy = adata_ood[adata_ood.obs["control"]].copy()
    pred_train = mean_model(adata_pred_train, adata_control_copy)
    conditions = [cond] * pred_train.shape[0]

    obs_data = pd.DataFrame({
        'condition': conditions
    })
    adata_pred_train = sc.AnnData(X=pred_train, obs=obs_data)
    cfpp.project_pca(query_adata=adata_pred_train, ref_adata=adata_ref)
    
    ood_data_target_decoded_predicted[cond] = adata_pred_train.X
    ood_data_target_encoded_predicted[cond] = adata_pred_train.obsm["X_pca"]
    del adata_pred_train
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs[cond_name] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs[cond_name] == cond].X.toarray()
    
del adata_ood
del adata_ref

# Intersection 
train_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
    }

del adata_train

def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train_vars]]

# Compute metrics 
ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, train_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, train_deg_dict)

ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics, ood_data_target_encoded, ood_data_target_encoded_predicted
)
ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)
deg_train_metrics = jax.tree_util.tree_map(compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted)

pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(save_path, f"mean_train_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(save_path, f"mean_train_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_train_metrics).to_csv(os.path.join(save_path, f"mean_train_metrics_deg_{split}.csv"))
