import scanpy as sc
import cfp
import os
import random
import pickle
import pandas as pd

data_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc"
adata=sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_new_cytokine_processed.h5ad")

annotations = pd.read_csv("/lustre/groups/ml01/workspace/ten_million/data/data_2024_12_16/new_cell_type_annotations.csv", index_col=0)
adata.obs["cell_type_new"] = annotations["cell_type_new"] 
adata.obs["donor_cell_type"] = adata.obs.apply(lambda x: x["donor"] + "_" + x["cell_type_new"], axis=1)

cfp.preprocessing.centered_pca(adata, n_comps=30, keep_centered_data=False)
adata.write("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")