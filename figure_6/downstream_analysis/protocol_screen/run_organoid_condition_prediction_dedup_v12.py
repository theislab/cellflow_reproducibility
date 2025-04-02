import warnings

warnings.filterwarnings("ignore")

import os

import matplotlib.pyplot as plt
import numpy as np
import jax.tree as jt
import jax.numpy as jnp
import cloudpickle
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, vstack
from plotnine import *
import itertools
import tqdm

from cfp.metrics import compute_scalar_mmd
import cfp


PLOT_DIR = "/home/fleckj/projects/cellflow/plots/organoid_cond_search/predictions/"
RESULTS_DIR = "/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/"
FULL_DATA_PATH = f"{DATA_DIR}/organoids_combined_full.h5ad"
adata = sc.read_h5ad(FULL_DATA_PATH)

#### Define trial name ####
# TRIAL_NAME = sys.argv[1]
TRIAL_NAME = "cellflow_0a37dcb9"
RESULTS_DIR = f"{RESULTS_DIR}/{TRIAL_NAME}/"

#### Load predictions ####
adata_v1 = sc.read(f"{RESULTS_DIR}/v1/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")
adata_v2 = sc.read(f"{RESULTS_DIR}/v2/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")

wknn_v1 = cloudpickle.load(
    open(f"{RESULTS_DIR}/v1/wknn/organoid_cond_preds_{TRIAL_NAME}_wknn_full.pkl", "rb")
)
wknn_v2 = cloudpickle.load(
    open(f"{RESULTS_DIR}/v2/wknn/organoid_cond_preds_{TRIAL_NAME}_wknn_full.pkl", "rb")
)

meta_v1 = pd.read_csv(f"{RESULTS_DIR}/v1/condition_meta_annot.tsv", sep="\t")
meta_v2 = pd.read_csv(f"{RESULTS_DIR}/v2/condition_meta_annot.tsv", sep="\t")

#### Join runs ####
meta_all = pd.concat([meta_v1, meta_v2], axis=0)
meta_all[meta_all == 0] = np.nan
meta_all_dedup = (
    meta_all.iloc[:, [0, 1, *list(range(4, 36))]]
    .set_index(["condition", "condition_new"])
    .drop_duplicates()
    .reset_index()
)

meta_all_dedup.to_csv(
    f"{RESULTS_DIR}/v1+2/condition_meta_annot.tsv", sep="\t", index=False
)
selection_v1 = adata_v1.obs["condition"].isin(meta_all_dedup["condition"])
adata_v1_dedup = adata_v1[selection_v1]

selection_v2 = adata_v2.obs["condition"].isin(
    meta_all_dedup["condition"]
) & ~adata_v2.obs["condition"].isin(adata_v1_dedup.obs["condition"])
adata_v2_dedup = adata_v2[selection_v2]

selection_obs = (adata_v1.obs["dataset"] == "observed") & (
    adata_v1.obs["condition"] != "CTRL"
)
adata_obs = adata_v1[selection_obs]

adata_v12 = ad.concat(
    {"v1": adata_v1_dedup, "v2": adata_v2_dedup, "obs": adata_obs},
    join="outer",
    label="run",
    fill_value=0,
)

adata_v12.write(f"{RESULTS_DIR}/v1+2/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")

wknn_v1_dedup = csr_matrix(wknn_v1)[selection_v1.values, :]
wknn_v2_dedup = csr_matrix(wknn_v2)[selection_v2.values, :]
wknn_obs = csr_matrix(wknn_v1)[selection_obs.values, :]

wknn_v12 = vstack([wknn_v1_dedup, wknn_v2_dedup, wknn_obs])

cloudpickle.dump(
    wknn_v12,
    open(f"{RESULTS_DIR}/v1+2/organoid_cond_preds_{TRIAL_NAME}_wknn_full.pkl", "wb"),
)

#### New annotations ####
adata_ref = sc.read(
    f"/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3.1umap_common_hv2k_wknn.h5ad"
)

selection_ref = adata_ref.obs["Region"] != "Forebrain"
adata_ref_use = adata_ref[selection_ref]
wknn_use = wknn_v12[:, selection_ref.values]
adata_ref_use.uns["wknn"] = wknn_use

cfp.pp.transfer_labels(adata_v12, adata_ref_use, label_key="Region", wknn_key="wknn")

selection_ref = ~adata_ref.obs["Subregion"].isin(["Forebrain", "Telencephalon"])
adata_ref_use = adata_ref[selection_ref]
adata_ref_use.obs["Subregion"][
    adata_ref_use.obs["Subregion"].isin(["Midbrain ventral", "Midbrain dorsal"])
] = "Midbrain"
wknn_use = wknn_v12[:, selection_ref.values]
adata_ref_use.uns["wknn"] = wknn_use

cfp.pp.transfer_labels(adata_v12, adata_ref_use, label_key="Subregion", wknn_key="wknn")

adata_v12.write_h5ad(f"{RESULTS_DIR}/v1+2/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")

adata_v12.obs.to_csv(
    f"{RESULTS_DIR}/v1+2/organoid_cond_preds_{TRIAL_NAME}_full_meta.tsv", sep="\t"
)


#### Join dists to reference ####
dists_v1 = pd.read_csv(
    f"{RESULTS_DIR}/v1/cluster_dists/organoid_cond_preds_{TRIAL_NAME}_dist_results.tsv",
    sep="\t",
)
dists_v2 = pd.read_csv(
    f"{RESULTS_DIR}/v2/cluster_dists/organoid_cond_preds_{TRIAL_NAME}_dist_results.tsv",
    sep="\t",
)

dists_v1_dedup = dists_v1[dists_v1["condition"].isin(meta_all_dedup["condition"])]
dists_v2_dedup = dists_v2[
    dists_v2["condition"].isin(meta_all_dedup["condition"])
    & ~dists_v2["condition"].isin(dists_v1_dedup["condition"])
]
dists_obs = dists_v1[dists_v1["dataset"] == "observed"]

dists_v12 = pd.concat([dists_v1_dedup, dists_v2_dedup, dists_obs], axis=0).iloc[:, 2:]

dists_v12.to_csv(
    f"{RESULTS_DIR}/v1+2/organoid_cond_preds_{TRIAL_NAME}_dist_results.tsv",
    sep="\t",
    index=False,
)

# %%
