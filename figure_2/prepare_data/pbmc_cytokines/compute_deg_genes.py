import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
import pandas as pd
import numpy as np
import pickle

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/adata_hvg2000_LV.h5ad")
annotations = pd.read_csv("/lustre/groups/ml01/workspace/ten_million/data/data_2024_12_16/new_cell_type_annotations.csv", index_col=0)
adata.obs["cell_type_new"] = annotations["cell_type_new"] 
adata.obs["donor_cell_type"] = adata.obs.apply(lambda x: x["donor"] + "_" + x["cell_type_new"], axis=1)
adata.X = adata.layers["counts"]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.obs["donor_cell_type_pert"] = adata.obs.apply(lambda x: x["donor"] + "_" + x["cell_type_new"] + "_" + x["cytokine"], axis=1)
conds = adata.obs["donor_cell_type_pert"].value_counts()
conds_to_keep = list(conds[conds>=50].index)
def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):
    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        adata_cov = adata[(adata.obs[covariate] == cov_cat)&(adata.obs["cell_type_new"]!="Platelet")]
        if len(adata_cov.obs["cytokine"].unique()) < 2:
            continue
        
        if adata_cov.n_obs == 0:
            continue

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False,
            method="wilcoxon",
        )
        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[f"{cov_cat}_{group}"] = de_genes[group].tolist()
    adata.uns[key_added] = gene_dict
    if return_dict:
        return gene_dict


deg_genes = rank_genes_groups_by_cov(
    adata[adata.obs["donor_cell_type_pert"].isin(conds_to_keep)],
    groupby="cytokine",
    control_group="PBS",
    covariate="donor_cell_type",
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=True,
)


with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs_2.pkl", "wb") as pickle_file:
    pickle.dump(deg_genes, pickle_file)

