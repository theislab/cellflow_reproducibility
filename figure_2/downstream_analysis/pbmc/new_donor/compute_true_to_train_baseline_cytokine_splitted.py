import scanpy as sc
import pandas as pd
from cellflow.metrics import compute_e_distance
import sys


donor = sys.argv[1]
adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")

test_cytokines = ['ADSF',
 'BAFF',
 'CD27L',
 'FasL',
 'IFN-gamma',
 'IFN-omega',
 'IL-1Ra',
 'IL-32-beta',
 'M-CSF',
 'OX40L']

train_cytokines = adata.obs["cytokine"].unique()
train_cytokines = [cyto for cyto in train_cytokines if cyto not in test_cytokines]

df = pd.DataFrame(columns=["donor", "test_cytokine", "reference_cyto", "e_distance"])
i=0
adata_donor = adata[adata.obs["donor"]==donor]
for cyto_1 in test_cytokines:
    for cyto_2 in train_cytokines:
        df.loc[i,:] = [donor, cyto_1, cyto_2, compute_e_distance(adata_donor[adata_donor.obs["cytokine"]==cyto_1].obsm["X_pca"], adata_donor[adata_donor.obs["cytokine"]==cyto_2].obsm["X_pca"])]
        i+=1
    print(f"{cyto_1} done.")

df.to_csv(f"/lustre/groups/ml01/workspace/ot_perturbation/data/true_to_train_baseline_cytokine_{donor}.csv")