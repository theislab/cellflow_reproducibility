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


def get_embedding(embedding: str) -> dict[str, np.ndarray]:
    if embedding == "esm2":
        adata_tmp = sc.read_h5ad("/home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata/adata_train_pca_50_split_0.h5ad")
        return adata_tmp.uns["esm2"]
    if embedding == "nargab":
        with open('/lustre/groups/ml01/workspace/ot_perturbation/data/embeddings/gene_nargab.pkl', 'rb') as f:
            nargab_emb = pickle.load(f)
            return nargab_emb
    else:
        raise ValueError("Embedding not supported")


def get_mask(x, y):
    return x[:, [gene in y for gene in adata_train.var_names]]

def get_train_embeddings(adata_train: ad.AnnData, embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    conds = adata_train.obs.drop_duplicates(subset=["condition"])
    emb_vectors = {}
    for _,row in conds.iterrows():
        gene_1 = row["gene_1"]
        gene_2  = row["gene_2"]
        if gene_1=="ctrl" and gene_2=="ctrl":
            continue
        elif gene_1 != "ctrl" and gene_2!= "ctrl":
            emb_vectors[(gene_1, gene_2)] = (embeddings[gene_1] + embeddings[gene_2])/2.0
        elif gene_1 != "ctrl" and gene_2=="ctrl":
            emb_vectors[(gene_1, gene_2)] = embeddings[gene_1]
        elif gene_1=="ctrl" and gene_2!="ctrl":
            emb_vectors[(gene_1, gene_2)] = embeddings[gene_2]
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






DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/norman_new"
dfs = []
for split in range(5):

    adata_train_path = os.path.join(DATA_DIR, f"adata_train_pca_50_split_{split}.h5ad")
    adata_test_path = os.path.join(DATA_DIR, f"adata_test_pca_50_split_{split}.h5ad")
    adata_ood_path = os.path.join(DATA_DIR, f"adata_ood_pca_50_split_{split}.h5ad")

    adata_train = sc.read(adata_train_path)
    adata_test = sc.read(adata_test_path)
    adata_ood = sc.read(adata_ood_path)

    ood_conds = adata_ood.obs.drop_duplicates()

    embeddings = get_embedding("esm2")
    train_embeddings = get_train_embeddings(adata_train, embeddings)
    conds = adata_ood.obs.drop_duplicates()
    preds = {}

    for _,row in conds.iterrows():    
        if row["condition"] == "ctrl":
            continue
        elif row["gene_2"] =="ctrl":
            emb_0 = row["gene_1"]
            closest_emb = find_closest_emb(embeddings[emb_0], train_embeddings)
        elif row["gene_1"] =="ctrl":
            emb_0 = row["gene_2"]
            closest_emb = find_closest_emb(embeddings[emb_0], train_embeddings)
        else:
            emb_0 = (embeddings[row["gene_1"]] + embeddings[row["gene_2"]])/2.0
            closest_emb = find_closest_emb(emb_0, train_embeddings)
        gene1, gene2 = row["gene_1"], row["gene_2"]
        preds[row["condition"]] = adata_train[adata_train.obs["condition"]==f"{closest_emb[0]}+{closest_emb[1]}"].X.toarray()



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
    df_all["model"] = "closest_embedding"
    dfs.append(df_all)


df_final = pd.concat(dfs, axis=0)
df_final.to_csv("/lustre/groups/ml01/workspace/ot_perturbation/data/norman_2/closest_embedding/norman_results_all.csv")