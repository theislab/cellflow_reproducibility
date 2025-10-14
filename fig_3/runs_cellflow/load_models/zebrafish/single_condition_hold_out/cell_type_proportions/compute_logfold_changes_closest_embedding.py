import pandas as pd
import numpy as np
import scanpy as sc
import os
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca
import sys
import anndata as ad
import pandas as pd

from typing import Any, Tuple


def get_train_embeddings(adata_same_timepoint: ad.AnnData, embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    conds = adata_same_timepoint.obs.drop_duplicates(subset="condition")
    emb_vectors = {}
    for _,row in conds.iterrows():
        g_1 = row["gene_target_1"]
        g_2  = row["gene_target_2"]
        if g_1=="control" and g_2=="control":
            continue
        elif g_1 != "control" and g_2!= "control":
            emb_vectors[(g_1, g_2)] = (embeddings[g_1] + embeddings[g_2])/2.0
        elif g_1 != "control" and g_2=="control":
            emb_vectors[(g_1, g_2)] = embeddings[g_1]
        elif g_1=="control" and g_2!="control":
            emb_vectors[(g_1, g_2)] = embeddings[g_2]
    return emb_vectors

def find_closest_embedding(emb_0: np.ndarray, reference_embeddings: dict[str, np.ndarray]) -> Tuple[str, ...]:
    closest_emb = None
    closest_dist = np.inf
    for ref, ref_emb in reference_embeddings.items():
        dist = np.sum((emb_0-ref_emb)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_emb = ref
    return closest_emb

if __name__=="__main__":
    ood_cond = sys.argv[1]

    tp = int(ood_cond.split("_")[-1])
    data_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/single_condition_f"
    out_dir = "/lustre/groups/ml01/workspace/cell_flow_zebrafish/results/single_condition/cell_type_proportion_closest_embedding"



    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    embeddings = adata.uns["gene_embeddings"]

    adata_ref_for_ct_error = adata[adata.obs["condition"]!=ood_cond]
    tp = int(ood_cond.split("_")[-1])
    gene_knockout = "_".join(ood_cond.split("_")[:-1])

    row = adata[adata.obs["condition"]==ood_cond].obs.drop_duplicates(subset="condition").iloc[0]
    adata_same_timepoint = adata[adata.obs["timepoint"]==tp]
    same_timepoint_embeddings = get_train_embeddings(adata_same_timepoint, embeddings)

    if row["condition"] == "control_control":
        pass
    elif row["gene_target_2"] == "control":
        emb_0 = ood_cond.split("_")[0]
        closest_emb = find_closest_embedding(embeddings[emb_0], same_timepoint_embeddings)
    else:
        emb_0 = (embeddings[ood_cond.split("_")[0]] + embeddings[ood_cond.split("_")[1]])/2.0
        closest_emb = find_closest_embedding(emb_0, same_timepoint_embeddings)


    adata_ood_pred = adata[adata.obs["gene_target"]==f"{closest_emb[0]}_{closest_emb[1]}"]

    if adata_ood_pred.n_obs > 10000:
        sc.pp.subsample(adata_ood_pred, n_obs=10000)

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


