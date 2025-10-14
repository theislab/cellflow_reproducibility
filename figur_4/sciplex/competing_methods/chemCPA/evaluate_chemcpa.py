import functools

import jax
import numpy as np
import scanpy as sc
import cfp.preprocessing as cfpp
from cfp.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
import sys


ood_split = sys.argv[1]

adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{ood_split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{ood_split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{ood_split}.h5ad"
print(1)
adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)
adata_pred_test = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/chemcpa/adata_chemcpa_split_{ood_split}_pred_test.h5ad")
adata_pred_ood = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/chemcpa/adata_chemcpa_split_{ood_split}_pred_ood.h5ad")
print(2)
adata_pred_ood.obs["dose_adapted"] = adata_pred_ood.obs.apply(lambda x: 0.0 if x["dose"]==1.0 else x["dose"], axis=1)
adata_pred_ood.obs["condition"] = adata_pred_ood.obs.apply(lambda x: x["cell_line"]+ "_"+x["perturbation"]+"_"+str(x["dose_adapted"]), axis=1)
adata_pred_test.obs["dose_adapted"] = adata_pred_test.obs.apply(lambda x: 0.0 if x["dose"]==1.0 else x["dose"], axis=1)
adata_pred_test.obs["condition"] = adata_pred_test.obs.apply(lambda x: x["cell_line"]+ "_"+x["perturbation"]+"_"+str(x["dose_adapted"]), axis=1)
print(3)

adata_ref = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/full_adata_with_splits.h5ad")
print(4)
cfpp.centered_pca(adata_ref, n_comps=20, method="rapids")
print(5)
cfpp.project_pca(query_adata=adata_pred_test, ref_adata=adata_ref)
print(6)
cfpp.project_pca(query_adata=adata_test, ref_adata=adata_ref)
test_data_target_encoded = {}
test_data_target_decoded = {}
test_data_target_encoded_predicted = {}
test_data_target_decoded_predicted = {}
for cond in adata_test.obs["condition"].cat.categories:
    if "Vehicle" in cond:
        continue
    test_data_target_encoded[cond] = adata_test[adata_test.obs["condition"] == cond].obsm["X_pca"]
    test_data_target_decoded[cond] = adata_test[adata_test.obs["condition"] == cond].X.toarray()
    test_data_target_decoded_predicted[cond] = adata_pred_test[adata_pred_test.obs["condition"] == cond].X
    test_data_target_encoded_predicted[cond] = adata_pred_test[adata_pred_test.obs["condition"] == cond].obsm["X_pca"]


cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if "Vehicle" in cond:
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].X
    ood_data_target_encoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].obsm["X_pca"]



test_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in test_data_target_decoded_predicted.keys()
}

ood_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
}

def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train.var_names]]


ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)
deg_ood_metrics = jax.tree_util.tree_map(compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted)

ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics, ood_data_target_encoded, ood_data_target_encoded_predicted
)

ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics, ood_data_target_decoded, ood_data_target_decoded_predicted
)


output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/results/chemcpa"

import os
import pandas as pd
split = ood_split
pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(output_dir, f"ood_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, f"ood_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_ood_metrics).to_csv(os.path.join(output_dir, f"ood_metrics_ood_{split}.csv"))

if split == "5":
    test_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, test_data_target_decoded_predicted, test_deg_dict)
    test_deg_target_decoded = jax.tree_util.tree_map(get_mask, test_data_target_decoded, test_deg_dict)
    deg_test_metrics = jax.tree_util.tree_map(compute_metrics, test_deg_target_decoded, test_deg_target_decoded_predicted)
    test_metrics_encoded = jax.tree_util.tree_map(
        compute_metrics, test_data_target_encoded, test_data_target_encoded_predicted
    )
    
    test_metrics_decoded = jax.tree_util.tree_map(
        compute_metrics_fast, test_data_target_decoded, test_data_target_decoded_predicted
    )

    pd.DataFrame.from_dict(test_metrics_encoded).to_csv(os.path.join(output_dir, f"test_metrics_encoded_{split}.csv"))
    pd.DataFrame.from_dict(test_metrics_decoded).to_csv(os.path.join(output_dir, f"test_metrics_decoded_{split}.csv"))
    pd.DataFrame.from_dict(deg_test_metrics).to_csv(os.path.join(output_dir, f"test_metrics_deg_{split}.csv"))

