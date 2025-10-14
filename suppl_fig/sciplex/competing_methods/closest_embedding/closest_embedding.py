import functools

import jax
import numpy as np
import scanpy as sc
import cellflow.preprocessing as cfpp
from cellflow.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
import sys
import anndata as ad
import scanpy as sc
import pandas as pd
from typing import Any, Tuple, Dict
import pandas as pd
import os
split = 5


adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"

adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)

adata_ref = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/full_adata_with_splits.h5ad")
cfpp.centered_pca(adata_ref, n_comps=20)

cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)



def get_train_embeddings(adata_train: ad.AnnData, embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    conds = adata_train.obs.drop_duplicates(subset="condition")
    emb_vectors = {}
    for _,row in conds.iterrows():
        d_1 = row["drug"]
        if d_1=="Vehicle":
            continue
        emb_vectors[d_1] = embeddings[d_1]
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


embeddings = adata_train.uns["ecfp_dict"]
train_embeddings = get_train_embeddings(adata_train, embeddings)
conds = adata_ood.obs.drop_duplicates(subset=["condition"])
preds = {}

for _,row in conds.iterrows():    
    if row["drug"] == "Vehicle":
        continue
    else:
        emb_0 = row["drug"]
        closest_emb = find_closest_emb(embeddings[emb_0], train_embeddings)
    print(closest_emb)
    adata_tmp = adata_train[(adata_train.obs["drug"]==closest_emb) & (adata_train.obs["cell_type"]==row["cell_type"])]
    assert adata_tmp.n_obs > 0
    adata_candidate = adata_tmp[(adata_tmp.obs["dose"]==row["dose"])]
    if adata_candidate.n_obs == 0:
        adata_candidate = adata_tmp[(adata_tmp.obs["logdose"]==row["logdose"]-1)]
        if adata_candidate.n_obs == 0:
            adata_candidate = adata_tmp[(adata_tmp.obs["logdose"]==row["logdose"]+1)]
    assert adata_candidate.n_obs > 0    
    n = min(adata_candidate.n_obs, 1500)
    preds[row["condition"]] = adata_candidate.X.toarray()[:n,:]
        

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


cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_ref)
cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_ref)



ood_data_target_encoded = {}
ood_data_target_decoded = {}
ood_data_target_encoded_predicted = {}
ood_data_target_decoded_predicted = {}
for cond in adata_ood.obs["condition"].cat.categories:
    if "Vehicle" in cond:
        continue
    src_str = list(adata_ood[adata_ood.obs["condition"] == cond].obs["cell_type"].unique())
    assert len(src_str) == 1
    
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


ood_metrics_encoded = jax.tree_util.tree_map(compute_metrics_fast, ood_data_target_encoded, ood_data_target_encoded_predicted)
ood_metrics_decoded = jax.tree_util.tree_map(compute_metrics_fast, ood_data_target_decoded, ood_data_target_decoded_predicted)
deg_ood_metrics = jax.tree_util.tree_map(compute_metrics_fast, ood_deg_target_decoded, ood_deg_target_decoded_predicted)

output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/results/closest_embedding"

import os
import pandas as pd

pd.DataFrame.from_dict(ood_metrics_encoded).to_csv(os.path.join(output_dir, f"ood_metrics_encoded_{split}.csv"))
pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, f"ood_metrics_decoded_{split}.csv"))
pd.DataFrame.from_dict(deg_ood_metrics).to_csv(os.path.join(output_dir, f"ood_metrics_ood_{split}.csv"))
