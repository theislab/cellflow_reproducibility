import scanpy as sc
import numpy as np
import functools
import jax
from cellflow.metrics import compute_metrics_fast, compute_mean_metrics, compute_metrics_fast
import cellflow.preprocessing as cfpp
import sys
import anndata as ad
from typing import Any, Tuple
import pandas as pd
import os

split = sys.argv[1]

adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_train_{split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_test_{split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_ood_{split}.h5ad"
adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)


def prepare_data(adata_train, adata_test, adata_ood):
    
    adata_tmp =  adata_train[adata_train.obs["Drug1"].drop_duplicates().index]
    ecfp_dict = {drug: adata_tmp[adata_tmp.obs["Drug1"]==drug].obsm["ecfp_drug_1"] for drug in adata_tmp.obs["Drug1"]}

    adata_tmp =  adata_train[adata_train.obs["Drug2"].drop_duplicates().index]
    ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug2"]==drug].obsm["ecfp_drug_2"] for drug in adata_tmp.obs["Drug2"]})

    adata_tmp =  adata_ood[adata_ood.obs["Drug1"].drop_duplicates().index]
    ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug1"]==drug].obsm["ecfp_drug_1"] for drug in adata_tmp.obs["Drug1"]})

    adata_tmp =  adata_ood[adata_ood.obs["Drug2"].drop_duplicates().index]
    ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug2"]==drug].obsm["ecfp_drug_2"] for drug in adata_tmp.obs["Drug2"]})

        
    adata_train.uns['ecfp_rep'] = ecfp_dict
    adata_test.uns['ecfp_rep'] = ecfp_dict
    adata_ood.uns['ecfp_rep'] = ecfp_dict
    return adata_train, adata_test, adata_ood


def get_train_embeddings(adata_train: ad.AnnData, embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    conds = adata_train.obs.drop_duplicates()
    emb_vectors = {}
    for _,row in conds.iterrows():
        d_1 = row["Drug1"]
        d_2  = row["Drug2"]
        if d_1=="control" and d_2=="control":
            continue
        elif d_1 != "control" and d_2!= "control":
            emb_vectors[(d_1, d_2)] = (embeddings[d_1] + embeddings[d_2])/2.0
        elif d_1 != "control" and d_2=="control":
            emb_vectors[(d_1, d_2)] = embeddings[d_1]
        elif d_1=="control" and d_2!="control":
            emb_vectors[(d_1, d_2)] = embeddings[d_2]
    return emb_vectors
        

def find_closest_emb(emb_0: np.ndarray, reference_embeddings: dict[str, np.ndarray]) -> Tuple[str, ...]:
    closest_emb = None
    closest_dist = np.inf
    for ref, ref_emb in reference_embeddings.items():
        dist = np.sum((emb_0-ref_emb)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_emb = ref
    return closest_emb

adata_train, adata_test, adata_ood = prepare_data(adata_train, adata_test, adata_ood)
embeddings = adata_train.uns["ecfp_rep"]
train_embeddings = get_train_embeddings(adata_train, embeddings)
conds = adata_ood.obs.drop_duplicates(subset=["condition"])
preds = {}

for _,row in conds.iterrows():    
    if row["condition"] == "control":
        continue
    elif row["Drug2"] =="control":
        emb_0 = row["Drug1"]
        closest_emb = find_closest_emb(embeddings[emb_0], train_embeddings)
    elif row["Drug1"] =="control":
        emb_0 = row["Drug2"]
        closest_emb = find_closest_emb(embeddings[emb_0], train_embeddings)
    else:
        emb_0 = (embeddings[row["Drug1"]] + embeddings[row["Drug2"]])/2.0
        closest_emb = find_closest_emb(emb_0, train_embeddings)
    d1, d2 = row["Drug1"], row["Drug2"]
    print(closest_emb)
    preds[row["condition"]] = adata_train[adata_train.obs["condition"]==f"{closest_emb[0]}+{closest_emb[1]}"].X.toarray()[:500,:]
        

all_data = []
conditions = []

for condition, array in preds.items():
    
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])

# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
obs_data = pd.DataFrame({
    'condition': conditions
})

# Create the Anndata object
adata_pred_ood = ad.AnnData(X=all_data_array, obs=obs_data)


adata_ref = ad.concat((adata_train, adata_test, adata_ood[adata_ood.obs["control"]==0]))
cfpp.centered_pca(adata_ref, n_comps=10)


cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if cond == "control":
        continue
    ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
    ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
    ood_data_target_decoded_predicted[cond] = adata_pred_ood.X
    ood_data_target_encoded_predicted[cond] = adata_pred_ood.obsm["X_pca"]


ood_deg_dict = {
    k: v
    for k, v in adata_train.uns["rank_genes_groups_cov_all"].items()
    if k in ood_data_target_decoded_predicted.keys()
}


def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train.var_names]]


ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)

deg_ood_metrics = jax.tree_util.tree_map(compute_metrics_fast, ood_deg_target_decoded, ood_deg_target_decoded_predicted)


ood_metrics_encoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_encoded, ood_data_target_encoded_predicted
)

ood_metrics_decoded = jax.tree_util.tree_map(
    compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted
)


output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/results/closest_embedding"

import os
import pandas as pd

pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(output_dir, f"ood_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, f"ood_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_ood_metrics).to_csv(os.path.join(output_dir, f"ood_metrics_ood_{split}.csv"))
