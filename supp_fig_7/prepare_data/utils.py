import os, sys
import scanpy as sc
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from scipy.sparse import csr_matrix, vstack
import random

from cfp.metrics import compute_metrics, compute_metrics_fast

def compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))

def reconstruct_data(embedding: np.ndarray, projection_matrix: np.ndarray, mean_to_add: np.ndarray) -> np.ndarray:
    return np.matmul(embedding, projection_matrix.T) + mean_to_add

def rank_genes_groups_by_cov(
    adata,
    groupby,
    covariate,
    control_group,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):
    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
        )
        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()
    adata.uns[key_added] = gene_dict
    if return_dict:
        return gene_dict

def get_DE_genes(adata, by='condition', covariate='cell_type'):
    adata.obs.loc[:, "control"] = adata.obs.gene.apply(lambda x: 1 if x == "NT" else 0)
    adata.obs = adata.obs.astype("category")
    rank_genes_groups_by_cov(
        adata,
        groupby=by,
        covariate=covariate,
        control_group="NT",
        n_genes=50,
        key_added="rank_genes_groups_cov_all",
    )
    return adata

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)

def data_preprocessing(adata, hvg, pert_threshold=100, mixscape_threshold=0.5, success=False, pert_key = 'gene', control_id = 'NT'):
    import pertpy as pt
    
    set_seed(42)
    pert_keep = list(adata.obs[pert_key].unique())
    while control_id in pert_keep:
        pert_keep.remove(control_id)
    genes_perturb = list(set(pert_keep).intersection(set(adata.var_names)))
    print(set(pert_keep).difference(set(genes_perturb)))
    adata = adata[adata.obs[pert_key].isin(genes_perturb + [control_id]), :]

    pert_cell_ids = []
    adata_pert = adata.copy()
    for ct in adata_pert.obs['cell_type'].unique():
        for pw in adata_pert.obs['pathway'].unique():
            subset = adata_pert[(adata_pert.obs['cell_type'] == ct) & (adata_pert.obs['pathway'] == pw),]
            sc.pp.normalize_total(subset)
            sc.pp.log1p(subset)
            sc.pp.highly_variable_genes(subset, subset=True)
            sc.pp.pca(subset)
            mixscape_identifier = pt.tl.Mixscape()
            mixscape_identifier.perturbation_signature(subset, pert_key, control_id)
            mixscape_identifier.mixscape(adata = subset, control = control_id, labels=pert_key, layer='X_pert')
            pert_cell_ids += subset[(subset.obs.mixscape_class_p_ko >= mixscape_threshold) | (subset.obs[pert_key]==control_id)].obs_names.tolist()

    adata = adata[pert_cell_ids, :]
    print(f'filtered {adata_pert.n_obs} cells to {adata.n_obs} cells due to low signal')
    # Keep just perturbations with more than 'pert_threshold' ocurrences AGAIN after Mixscape
    perturbation_counts = adata.obs[pert_key].value_counts()
    perturbations_to_keep = perturbation_counts[perturbation_counts >= pert_threshold].index.tolist()
    adata = adata[adata.obs[pert_key].isin(perturbations_to_keep), :]
    print(f'filtered to {adata.n_obs} cells due to low counts')

    return adata
