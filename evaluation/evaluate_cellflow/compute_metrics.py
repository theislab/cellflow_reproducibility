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

print("Start")
split = sys.argv[1]

# Read the reference dataset plus centered PCA 
adata_ref = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/full_adata_with_splits.h5ad")
print("Centered pca")
cfpp.centered_pca(adata_ref, n_comps=20, keep_centered_data=False)

# Folders of real and generated data 
path_to_splits = Path("/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5")
path_to_generated = Path("/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data")

# Read the real datasets 
adata_train_path = path_to_splits / f"adata_train_split_{split}.h5ad"
adata_ood_path = path_to_splits / f"adata_ood_split_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_ood = sc.read_h5ad(adata_ood_path)

# Read the generated datasets 
adata_pred_ood = sc.read(path_to_generated / f"adata_ood_split_{split}.h5ad")
adata_pred_ood.X = adata_pred_ood.layers["X_recon_pred"]
print("Read ref")

# Projected PCA of the OOD datasets on the reference
print("Projected pca")
cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)

# Collect observations 
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}

for cond in adata_ood.obs["perturbation_condition"].cat.categories:
    print(f"Condition {cond}")
    if "NT" in cond:
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["perturbation_condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["perturbation_condition"] == cond].X.toarray()
    ood_data_target_encoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["perturbation_condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["perturbation_condition"] == cond].layers["X_recon_pred"]
print("End")

ood_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
    }

adata_train_vars = adata_train.var_names
del adata_ref
del adata_train

def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train_vars]]

# Compute metrics 
ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)

print("Start evaluating the metrics")

ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics, ood_data_target_encoded, ood_data_target_encoded_predicted
)
ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)
deg_ood_metrics = jax.tree_util.tree_map(compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted)

output_dir = f"/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/split{split}"

pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(output_dir, f"ood_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, f"ood_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_ood_metrics).to_csv(os.path.join(output_dir, f"ood_metrics_ood_{split}.csv"))
