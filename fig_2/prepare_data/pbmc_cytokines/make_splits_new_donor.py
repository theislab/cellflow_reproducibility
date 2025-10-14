import scanpy as sc
import sys
import random
import numpy as np
import cfp
import itertools
import os
import anndata as ad

def string_to_int(string, seed=None, mod=None):
    """
    Generate an integer from a string deterministically.
    
    Args:
        string (str): Input string to convert.
        seed (int, optional): A seed to make the hash consistent. Default is None.
        mod (int, optional): Modulo value to limit the integer range. Default is None.
    
    Returns:
        int: Deterministically generated integer.
    """
    # Use a seed to make the hash deterministic across runs if specified
    if seed is not None:
        random.seed(seed)
    hash_value = hash(string)
    # Optionally apply modulo for limiting range
    if mod is not None:
        return abs(hash_value) % mod
    return abs(hash_value)


ood_donor = sys.argv[1]

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_new_donor_processed.h5ad")
cytokines_to_impute = adata.uns['cytokines_to_impute']
cytokines_to_train_data = adata.uns['cytokines_to_train_data']

uns = adata.uns.copy()
adata_base = adata[adata.obs["donor"]!=ood_donor].copy()
adata_rest = adata[adata.obs["donor"]==ood_donor].copy()

adata_rest.uns["split_info"] = {}
unique_donors = list(adata.obs["donor"].unique())


d = {}

d["0"] = {}
d["0"]["cytokines_to_train_data"] = ["PBS"]
d["0"]['cytokines_to_impute'] = cytokines_to_impute
it = 0
for c in cytokines_to_train_data.values():
    for split in c:
        d[str(it)] = {}
        d[str(it)]["cytokines_to_train_data"] = split
        d[str(it)]["cytokines_to_impute"] = cytokines_to_impute
        
        adata_to_append = adata_rest[adata_rest.obs["cytokine"].isin(split)]
        adata_train = ad.concat((adata_base, adata_to_append))
        adata_train.uns = adata_base.uns.copy()
        adata_ood_perturbed = adata_rest[adata_rest.obs["cytokine"].isin(cytokines_to_impute)]
        adata_train.uns["split_info"] = d
        adata_ood_perturbed.uns["split_info"] = d

        os.makedirs(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}", exist_ok=True)
        adata_train.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}/adata_train_{ood_donor}.h5ad")
        adata_ood_perturbed.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}/adata_ood_{ood_donor}.h5ad")
        it+=1

os.makedirs(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}", exist_ok=True)
adata_train.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}/adata_train_{ood_donor}.h5ad")
adata_ood_perturbed.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{ood_donor}/{str(it)}/adata_ood_{ood_donor}.h5ad")


