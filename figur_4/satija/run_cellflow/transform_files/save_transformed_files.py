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
adata_path = sys.argv[2]
ckpt_path = sys.argv[3]
task = sys.argv[4]
save_path = sys.argv[5]

# Read data 
path_to_splits = Path(adata_path)
adata_train_path = path_to_splits / f"adata_train_{split}.h5ad"
adata_test_path = path_to_splits / f"adata_test_{split}.h5ad"
adata_ood_path = path_to_splits / f"adata_ood_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_test = sc.read_h5ad(adata_test_path)
adata_ood = sc.read_h5ad(adata_ood_path)

# Take controls 
adata_ood_ctrl = adata_ood[adata_ood.obs["control"]]

# Unique condition data frame 
if task=="ood_genes":
    covariate_data_ood = adata_ood[~adata_ood.obs["control"]].obs.drop_duplicates(subset=["perturbation_condition"])
else:
    covariate_data_ood = adata_ood[~adata_ood.obs["control"]].obs.drop_duplicates(subset=["condition"])

ckpt_path = Path(ckpt_path)
key = "cell_type" if task=="ood_cell_line" else "gene"  
cf = cfp.model.CellFlow.load(ckpt_path / f"satija_{key}_{split}_CellFlow.pkl")

if task=="ood_genes":
    preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="perturbation_condition", covariate_data=covariate_data_ood)
else:
    preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_ood)

# Collect the data together 
all_data = []
conditions = []
for condition, array in preds_ood.items():
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])
    
# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
if task=="ood_genes":
    obs_data = pd.DataFrame({
        'perturbation_condition': conditions
    })
else:
    obs_data = pd.DataFrame({
        'condition': conditions
    })

# Create the Anndata object
save_path = Path(save_path)
adata_ood_result = ad.AnnData(X=np.empty((len(all_data_array), adata_train.shape[1])), obs=obs_data)
adata_ood_result.obsm["X_pca_pred"] = all_data_array
cfpp.reconstruct_pca(query_adata=adata_ood_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
adata_ood_result.var = adata_ood.var.copy()
adata_ood_result.write(save_path / f"adata_ood_split_{split}.h5ad")

# # Predict on the test set 
# preds_test = cf.predict(adata=adata_test_ctrl, sample_rep="X_pca", condition_id_key="perturbation_condition", covariate_data=covariate_data_test)

# all_data = []
# conditions = []
# for condition, array in preds_test.items():
#     all_data.append(array)
#     conditions.extend([condition] * array.shape[0])
    
# # Stack all data vertically to create a single array
# all_data_array = np.vstack(all_data)

# # Create a DataFrame for the .obs attribute
# obs_data = pd.DataFrame({
#     'perturbation_condition': conditions
# })

# # Create the Anndata object
# adata_test_result = ad.AnnData(X=np.empty((len(all_data_array), 8265)), obs=obs_data)
# adata_test_result.obsm["X_pca_pred"] = all_data_array
# cfpp.reconstruct_pca(query_adata=adata_test_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
# adata_test_result.write(f"/lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/adata_test_split_{split}.h5ad")
