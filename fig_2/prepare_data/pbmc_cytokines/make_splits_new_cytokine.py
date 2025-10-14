import scanpy as sc
import sys
import random
import cfp
import itertools
import numpy as np

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


ood_cytokine = sys.argv[1]

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_new_cytokine_processed.h5ad")

uns = adata.uns.copy()
adata_base = adata[adata.obs["cytokine"]!=ood_cytokine].copy()
adata_rest = adata[adata.obs["cytokine"]==ood_cytokine].copy()

adata_rest.uns["split_info"] = {}
adata_rest.uns["split_info"] = {}
unique_donors = list(adata.obs["donor"].unique())

rng = np.random.default_rng(string_to_int(ood_cytokine))
results = {}

results[0] = [[]]

for k in range(1, 11):
    combinations = list(itertools.combinations(unique_donors, k))
    rng.shuffle(combinations)  # Shuffle to ensure randomness
    unique_sets = set()

    res = []
    for i in range(3):
        candidate = rng.choice(combinations)
        res.append(list(candidate))
    
    results[k] = res

res = []
for donor in unique_donors:
    other_donors = list(set(unique_donors) - set([donor]))
    res.append(other_donors)

results[11] = res

d = {}
it = 0
n_donors_to_impute = 0

for n_donor_to_train, splits in results.items():
    for split in splits:
        d[str(it)] = {}
        d[str(it)]["donors_to_train_data"] = split
        d[str(it)]["donors_to_impute"] = list(set(unique_donors) - set(split))
        it+=1
adata_rest.uns["split_info"]  = d

adata_base.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_base_{ood_cytokine}.h5ad")
adata_rest.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_rest_{ood_cytokine}.h5ad")

