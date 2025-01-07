import os
import cfp
import scanpy as sc
import numpy as np
import optax
import hydra 
from omegaconf import DictConfig, OmegaConf
from functools import partial
from cfp.training import Metrics, PCADecodedMetrics
from cfp.metrics import compute_metrics, compute_metrics_fast
from sklearn.metrics import r2_score
import datetime
import yaml
import jax.numpy as jnp
import torch
from cfp.data._dataloader import ValidationSampler
import pandas as pd
import jax.tree_util as jtu
import cfp.preprocessing as cfpp

import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path 
import sys

# Read the split 
split = sys.argv[1]

# Read data 
path_to_splits = Path("/lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5")
adata_train_path = path_to_splits / f"adata_train_split_{split}.h5ad"
adata_test_path = path_to_splits / f"adata_test_split_{split}.h5ad"
adata_ood_path = path_to_splits / f"adata_ood_split_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_test = sc.read_h5ad(adata_test_path)
adata_ood = sc.read_h5ad(adata_ood_path)

# Take controls 
adata_ood_ctrl = adata_ood[adata_ood.obs["control"]]
adata_test_ctrl = adata_test[adata_test.obs["control"]]

covariate_data_ood = adata_ood[~adata_ood.obs["control"]].obs.drop_duplicates(subset=["perturbation_condition"])
covariate_data_test = adata_test[~adata_test.obs["control"]].obs.drop_duplicates(subset=["perturbation_condition"])

cf = cfp.model.CellFlow.load(f"/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/ckpt_split_{split}/satija_gene_ood_{split}_CellFlow.pkl")

preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="perturbation_condition", covariate_data=covariate_data_ood)

# Collect the data together 
all_data = []
conditions = []
for condition, array in preds_ood.items():
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])
    
# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
obs_data = pd.DataFrame({
    'perturbation_condition': conditions
})

# Create the Anndata object
adata_ood_result = ad.AnnData(X=np.empty((len(all_data_array), 8265)), obs=obs_data)
adata_ood_result.obsm["X_pca_pred"] = all_data_array
cfpp.reconstruct_pca(query_adata=adata_ood_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
adata_ood_result.write(f"/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/adata_ood_split_{split}.h5ad")

# Predict on the test set 
preds_test = cf.predict(adata=adata_test_ctrl, sample_rep="X_pca", condition_id_key="perturbation_condition", covariate_data=covariate_data_test)

all_data = []
conditions = []
for condition, array in preds_test.items():
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])
    
# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
obs_data = pd.DataFrame({
    'perturbation_condition': conditions
})

# Create the Anndata object
adata_test_result = ad.AnnData(X=np.empty((len(all_data_array), 8265)), obs=obs_data)
adata_test_result.obsm["X_pca_pred"] = all_data_array
cfpp.reconstruct_pca(query_adata=adata_test_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
adata_test_result.write(f"/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/adata_test_split_{split}.h5ad")
