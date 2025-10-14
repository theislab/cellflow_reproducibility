import os
import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import rapids_singlecell as rsc
import scvi


#### Load datasets ####
combined_adata = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/organoids_combined_full.h5ad"
)

split_meta = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/organoid_holdout_splits.tsv",
    sep="\t",
)

RESULTS_DIR = "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/"


split_tasks = split_meta["split_task"].unique().tolist()

# Make the splits from the combined dataset
for task in split_tasks:
    task_meta = split_meta[split_meta["split_task"] == task]
    datasets = task_meta["dataset"].unique().tolist()

    for ds in datasets:
        ds_meta = task_meta[task_meta["dataset"] == ds]
        splits = ds_meta["holdout_combination"].unique().tolist()

        for split in splits:
            this_split_meta = ds_meta[ds_meta["holdout_combination"] == split]
            holdout_conds = this_split_meta["holdout_condition"].tolist()

            adata_train = combined_adata[
                ~combined_adata.obs["condition"].isin(holdout_conds)
            ].copy()

            test_idx = combined_adata.obs["condition"].isin(holdout_conds)
            test_ctrl_idx = np.where(combined_adata.obs["CTRL"])[0]
            np.random.seed(111)
            test_ctrl_idx_subs = np.random.choice(test_ctrl_idx, 3000, replace=False)

            adata_test = combined_adata[
                test_idx
                | np.isin(np.arange(combined_adata.shape[0]), test_ctrl_idx_subs)
            ].copy()

            split_dir = f"{RESULTS_DIR}/{task}/{ds}/{split}/"
            os.makedirs(split_dir, exist_ok=True)

            adata_train.write(f"{split_dir}/adata_train.h5ad")
            adata_test.write(f"{split_dir}/adata_test.h5ad")
