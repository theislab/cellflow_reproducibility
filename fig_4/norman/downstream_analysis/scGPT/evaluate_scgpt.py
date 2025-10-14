import jax
import scanpy as sc
import cfp.preprocessing as cfpp
from cfp.metrics import compute_mean_metrics, compute_metrics
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ood_split', type=int, required=True)
parser.add_argument('--scenario', type=str, required=True)
args = parser.parse_args()

ood_split = args.ood_split
scenario = args.scenario

basepath = "/lustre/groups/ml01/workspace/leander.dony/projects/cellflow/250302_scGPT/"

adata_all = sc.read(os.path.join(basepath, "adatas", "adata_all.h5ad"))
cfpp.centered_pca(adata_all, n_comps=10)

results = {
    "comboseen0":{"mean": {}}, 
    "comboseen1":{"mean": {}}, 
    "comboseen2":{"mean": {}}, 
    "unseensingle":{"mean": {}}
    }

for ood_setting, res in results.items():
    adata_ood = sc.read(os.path.join(basepath, "adatas", f"norman_{scenario}_seed{ood_split}_test_{ood_setting}_truth.h5ad"))
    adata_pred_ood = sc.read(os.path.join(basepath, "adatas", f"norman_{scenario}_seed{ood_split}_test_{ood_setting}_pred.h5ad"))

    cfpp.project_pca(query_adata=adata_pred_ood, ref_adata=adata_all)
    cfpp.project_pca(query_adata=adata_ood, ref_adata=adata_all)
    ood_data_target_encoded = {}
    ood_data_target_decoded = {}
    ood_data_target_encoded_predicted = {}
    ood_data_target_decoded_predicted = {}
    for cond in adata_ood.obs["condition"].cat.categories:
        ood_data_target_encoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].obsm["X_pca"]
        ood_data_target_decoded[cond] = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
        ood_data_target_decoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].X
        ood_data_target_encoded_predicted[cond] = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].obsm["X_pca"]
        res[cond] = {}

    ood_deg_dict = {
        k: v
        for k, v in adata_all.uns["rank_genes_groups_cov_all"].items()
        if k in ood_data_target_decoded_predicted.keys()
    }

    def get_mask(x, y):
        return x[:, [gene in y for gene in adata_all.var_names]]

    ood_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, ood_data_target_decoded_predicted, ood_deg_dict)
    ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)

    metrics = jax.tree_util.tree_map(compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted)
    for k, v in metrics.items():
        res[k].update({f'deg_ood_{k}': v for k, v in v.items()})
    res["mean"].update(compute_mean_metrics(metrics, prefix="deg_ood_"))

    metrics = jax.tree_util.tree_map(
        compute_metrics, ood_data_target_encoded, ood_data_target_encoded_predicted
    )
    for k, v in metrics.items():
        res[k].update({f'encoded_ood_{k}': v for k, v in v.items()})
    res["mean"].update(compute_mean_metrics(metrics, prefix="encoded_ood_"))

    metrics = jax.tree_util.tree_map(
        compute_metrics, ood_data_target_decoded, ood_data_target_decoded_predicted
    )
    for k, v in metrics.items():
        res[k].update({f'decoded_ood_{k}': v for k, v in v.items()})
    res["mean"].update(compute_mean_metrics(metrics, prefix="decoded_ood_"))
    
    pd.DataFrame(res).T.to_csv(os.path.join(basepath, "metrics", f"metrics_{scenario}_seed{ood_split}_{ood_setting}.csv"))
