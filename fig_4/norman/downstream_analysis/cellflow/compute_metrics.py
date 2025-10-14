
import scanpy as sc
import jax
import os
from cellflow.metrics import compute_metrics, compute_mean_metrics, compute_metrics_fast
import cellflow.preprocessing as cfpp
import anndata as ad
import pandas as pd
import numpy as np
import sys
import pickle
import anndata as ad
import pandas as pd
from typing import Any, Tuple, Dict
import sys

split = sys.argv[1]

DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/norman_new"


adata_train_path = os.path.join(DATA_DIR, f"adata_train_pca_50_split_{split}.h5ad")
adata_test_path = os.path.join(DATA_DIR, f"adata_test_pca_50_split_{split}.h5ad")
adata_ood_path = os.path.join(DATA_DIR, f"adata_ood_pca_50_split_{split}.h5ad")

adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)

ood_conds = adata_ood.obs.drop_duplicates()

files = os.listdir(f"/lustre/groups/ml01/workspace/ot_perturbation/results/norman/cellflow_new/out/") # added _new
file = [f for f in files if f"with_predictions_{split}" in f][0]
adata_pred_ood = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/results/norman/cellflow_new/out/{file}") # added _new
adata_pred_ood.X = adata_pred_ood.layers["X_recon_pred"]

adata_ref = ad.concat((adata_train, adata_test, adata_ood[adata_ood.obs["control"]==0]))
cfpp.centered_pca(adata_ref, n_comps=10)


cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if cond == "ctrl":
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].X
    ood_data_target_encoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].obsm["X_pca"]


ood_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
}


def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train.var_names]]


ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)


ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_encoded, ood_data_target_encoded_predicted
)

ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)




adata_ref = ad.concat((adata_train, adata_test, adata_ood[adata_ood.obs["control"]==0]))
cfpp.centered_pca(adata_ref, n_comps=10)


cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if cond == "ctrl":
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_encoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].obsm["X_pca"]


ood_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
}

ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)


ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_encoded, ood_data_target_encoded_predicted
)

ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)

ood_metrics_deg = jax.tree_util.tree_map(
    compute_metrics_fast, ood_deg_target_decoded, ood_deg_target_decoded_predicted
)


df_enc = pd.DataFrame.from_dict(ood_metrics_encoded,orient="index")#.to_csv(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}.csv"))
df_dec = pd.DataFrame.from_dict(ood_metrics_decoded,orient="index")
df_deg = pd.DataFrame.from_dict(ood_metrics_deg,orient="index")

for col in df_enc.columns:
    df_enc[f"encoded_ood_{col}"] = df_enc[col]
    del df_enc[col]

for col in df_dec.columns:
    df_dec[f"decoded_ood_{col}"] = df_dec[col]
    del df_dec[col]

for col in df_deg.columns:
    df_deg[f"deg_ood_{col}"] = df_deg[col]
    del df_deg[col]

df_enc["condition"] = df_enc.index
df_all = pd.concat((df_enc, df_dec, df_deg), axis=1)
df_cat = pd.concat((adata_train.obs.drop_duplicates(subset="condition"),adata_ood.obs.drop_duplicates(subset="condition")))
cond_to_cat = df_cat.set_index("condition")["subgroup"].to_dict()
df_all["subgroup"] = df_all["condition"].map(cond_to_cat)
df_all["model"] = "cellflow"


df_all.to_csv(f"/lustre/groups/ml01/workspace/ot_perturbation/data/norman_2/cellflow/norman_results_all_{split}_new.csv") # added _new