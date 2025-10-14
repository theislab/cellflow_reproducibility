import scanpy as sc
import sys
import cfp

ood_cytokine = sys.argv[1]

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_processed.h5ad")

uns = adata.uns.copy()
adata_train = adata[adata.obs["cytokine"]!=ood_cytokine].copy()
adata_ood_perturbed = adata[adata.obs["cytokine"]==ood_cytokine].copy()

adata_train.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/splits/adata_train_{ood_cytokine}.h5ad")
adata_ood_perturbed.write(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/splits/adata_ood_{ood_cytokine}.h5ad")

