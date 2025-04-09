import scanpy as sc
import pandas as pd
from cellflow.metrics import compute_e_distance

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

donors = adata.obs["donor"].unique()

df = pd.DataFrame(columns=["cytokine", "donor_1", "donor_2", "e_distance"])
i=0
for cyto in test_cytokines:
    adata_cyto = adata[adata.obs["cytokine"]==cyto]
    for donor1 in donors:
        for donor2 in donors:
            df.loc[i,:] = [cyto, donor1, donor2, compute_e_distance(adata_cyto[adata_cyto.obs["donor"]==donor1].obsm["X_pca"], adata_cyto[adata_cyto.obs["donor"]==donor2].obsm["X_pca"])]
            i+=1

df.to_csv("/lustre/groups/ml01/workspace/ot_perturbation/data/true_to_train_baseline_donor.csv")