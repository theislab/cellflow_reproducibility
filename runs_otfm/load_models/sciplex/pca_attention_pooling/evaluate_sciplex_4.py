import functools
import jax
import numpy as np
import scanpy as sc
import cfp.preprocessing as cfpp
from cfp.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
import os
import pandas as pd

ood_split=split = 4


adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{ood_split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{ood_split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{ood_split}.h5ad"

adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)
adata_pred_test = sc.read(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex/pca_attention_pooling/adata_test_with_predictions_{split}.h5ad")
adata_pred_ood = sc.read(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex/pca_attention_pooling/adata_ood_with_predictions_{split}.h5ad")
adata_pred_ood.X = adata_pred_ood.layers["X_recon_pred"]
adata_pred_test.X = adata_pred_test.layers["X_recon_pred"]

adata_ref_ood = adata_ood[~adata_ood.obs["condition"].str.contains('Vehicle')].copy()
cfpp.centered_pca(adata_ref_ood, n_comps=10)

adata_ref_test = adata_test[~adata_test.obs["condition"].str.contains('Vehicle')].copy()
cfpp.centered_pca(adata_ref_test, n_comps=10)


cfpp.project_pca(query_adata=adata_pred_test, ref_adata=adata_ref_test)
cfpp.project_pca(query_adata=adata_test, ref_adata=adata_ref_test)
test_data_target_encoded = {}
test_data_target_decoded = {}
test_data_target_encoded_predicted = {}
test_data_target_decoded_predicted = {}
for cond in adata_test.obs["condition"].cat.categories:
    if "Vehicle" in cond:
        continue
    test_data_target_encoded[cond] = adata_test[adata_test.obs["condition"] == cond].obsm["X_pca"]
    test_data_target_decoded[cond] = adata_test[adata_test.obs["condition"] == cond].X.toarray()
    test_data_target_decoded_predicted[cond] = adata_pred_test[adata_pred_test.obs["condition"] == cond].layers["X_recon_pred"]
    test_data_target_encoded_predicted[cond] = adata_pred_test[adata_pred_test.obs["condition"] == cond].obsm["X_pca"]


cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref_ood)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref_ood)
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if "Vehicle" in cond:
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].layers["X_recon_pred"]
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

test_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, test_data_target_decoded_predicted, test_deg_dict)
test_deg_target_decoded = jax.tree_util.tree_map(get_mask, test_data_target_decoded, test_deg_dict)

deg_ood_metrics = jax.tree_util.tree_map(compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted)
deg_mean_ood_metrics = compute_mean_metrics(deg_ood_metrics, prefix="deg_ood_")

deg_test_metrics = jax.tree_util.tree_map(compute_metrics_fast, test_deg_target_decoded, test_deg_target_decoded_predicted)
deg_mean_test_metrics = compute_mean_metrics(deg_test_metrics, prefix="deg_test_")

ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics, ood_data_target_encoded, ood_data_target_encoded_predicted
)
mean_ood_metrics_encoded = compute_mean_metrics(ood_metrics_encoded, prefix="encoded_ood_")

ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)
mean_ood_metrics_decoded = compute_mean_metrics(ood_metrics_decoded, prefix="decoded_ood_")


test_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics_fast, test_data_target_encoded, test_data_target_encoded_predicted
)
mean_test_metrics_encoded = compute_mean_metrics(test_metrics_encoded, prefix="encoded_test_")

test_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, test_data_target_decoded, test_data_target_decoded_predicted
)
mean_test_metrics_decoded = compute_mean_metrics(test_metrics_decoded, prefix="decoded_test_")

output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/results/otfm/pca_mean_pooling"


split = ood_split
pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(output_dir, f"ood_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, f"ood_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(test_metrics_encoded).to_csv(os.path.join(output_dir, f"test_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(test_metrics_decoded).to_csv(os.path.join(output_dir, f"test_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_test_metrics).to_csv(os.path.join(output_dir, f"test_metrics_deg_{split}.csv"))
pd.DataFrame.from_dict(deg_ood_metrics).to_csv(os.path.join(output_dir, f"ood_metrics_ood_{split}.csv"))


