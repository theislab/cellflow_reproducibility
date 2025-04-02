import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import jax
import jax.tree as jt
import jax.numpy as jnp
import cloudpickle
import pandas as pd
import scanpy as sc
from plotnine import *
import itertools
import tqdm

from cfp.metrics import compute_e_distance, compute_scalar_mmd
import cfp


def pairwise_distances(x, y):
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


@jax.jit
def compute_e_distance_fast(x, y) -> float:
    """Compute the energy distance as in Peidli et al."""
    sigma_X = pairwise_distances(x, x).mean()
    sigma_Y = pairwise_distances(y, y).mean()
    delta = pairwise_distances(x, y).mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_dists(x, y):
    edist = compute_e_distance_fast(x, y)
    return {
        "edist": float(edist),
    }


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
RESULTS_DIR = f"{RESULTS_DIR}/{TRIAL_NAME}/v1+2/"


#### Load predictions ####
adata_pred = sc.read(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")
cond_meta = pd.read_csv(f"{RESULTS_DIR}/condition_meta_annot.tsv", sep="\t")

#### Prepare full distributions ####
pred_conds = cond_meta["condition"][cond_meta["dataset"] != "observed"].unique()
ds_latents = {}
for dataset in ["neal", "fatima", "nadya"]:
    adata_ds = adata_pred[adata_pred.obs["dataset"] == dataset, :].copy()
    ds_latents[dataset] = {}
    for cond in tqdm.tqdm(pred_conds):
        cond_latent = adata_ds.obsm["X_latent"][adata_ds.obs["condition"] == cond, :]
        ds_latents[dataset][cond] = cond_latent

    cloudpickle.dump(
        ds_latents[dataset],
        open(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_{dataset}.pkl", "wb"),
    )


cloudpickle.dump(
    ds_latents, open(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full.pkl", "wb")
)


#### Compute e-distances to observed clusters ####
ds_latents = cloudpickle.load(
    open(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full.pkl", "rb")
)
adata_obs = adata_pred[adata_pred.obs["dataset"] == "observed", :]
obs_conds = adata_obs.obs["condition"].unique()

condition_dists = {}
for cond in tqdm.tqdm(obs_conds):
    print(cond)

    cond_latent = adata_obs[adata_obs.obs["condition"] == cond, :].obsm["X_latent"]
    cond_latent = jnp.array(cond_latent)
    cond_latent = cond_latent[
        np.random.choice(cond_latent.shape[0], 1000, replace=True), :
    ]

    dist_func = lambda x: compute_dists(x, cond_latent)
    dists = jt.map(dist_func, ds_latents)
    dists_df = pd.concat({k: pd.DataFrame(v).T for k, v in dists.items()}, axis=0)
    dists_df.index = dists_df.index.rename(["dataset", "condition"])
    dists_df.reset_index(inplace=True)
    dists_df["obs_condition"] = cond
    condition_dists[cond] = dists_df

condition_dist_df = pd.concat(condition_dists).reset_index(drop=True)
condition_dist_df.to_csv(
    f"{RESULTS_DIR}/organoid_cond_preds_obs_condition_dists.tsv",
    sep="\t",
    index=False,
)


#### Compute E-distance observed vs observed ####
obs_latents = {
    cond: adata_obs[adata_obs.obs["condition"] == cond, :].obsm["X_latent"]
    for cond in obs_conds
}

# Sizes
np.median(np.array([obs_latents[k].shape[0] for k in obs_latents.keys()]))

obs_dists = {}
for cond in tqdm.tqdm(obs_conds):
    print(cond)

    cond_latent = adata_obs[adata_obs.obs["condition"] == cond, :].obsm["X_latent"]
    cond_latent = jnp.array(cond_latent)
    cond_latent = cond_latent[
        np.random.choice(cond_latent.shape[0], 500, replace=True), :
    ]

    dist_func = lambda x: compute_dists(x, cond_latent)
    dists = jt.map(dist_func, obs_latents)
    dists_df = pd.DataFrame(dists).T
    dists_df.index = dists_df.index.rename("condition")
    dists_df.reset_index(inplace=True)
    dists_df["obs_condition"] = cond
    obs_dists[cond] = dists_df

obs_dist_df = pd.concat(obs_dists).reset_index(drop=True)
obs_dist_df.to_csv(
    f"{RESULTS_DIR}/organoid_obs_obs_dists.tsv",
    sep="\t",
    index=False,
)


#### Transfer cluster labels ####
adata_ref = sc.read(
    f"/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3.1umap_common_hv2k_wknn.h5ad"
)
wknn = cloudpickle.load(
    open(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_wknn_full.pkl", "rb")
)

adata_ref.uns["wknn"] = wknn
cfp.pp.transfer_labels(adata_pred, adata_ref, label_key="Clusters", wknn_key="wknn")

adata_pred.write(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")


#### Compute distances of transferred clusters to reference ####
adata_pred = sc.read(f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full.h5ad")
adata_ref = sc.read(
    f"/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3.1umap_common_hv2k_wknn.h5ad"
)

all_clusters = adata_pred.obs["Clusters_transfer"].unique().tolist()

cluster_latents = {}
for dataset in ["neal", "fatima", "nadya", "observed"]:
    adata_ds = adata_pred[adata_pred.obs["dataset"] == dataset, :].copy()
    ds_conds = adata_ds.obs["condition"].unique().tolist()
    cluster_latents[dataset] = {}

    for cluster in tqdm.tqdm(all_clusters):
        adata_cluster = adata_ds[adata_ds.obs["Clusters_transfer"] == cluster, :].copy()
        cluster_cond_latents = {
            c: adata_cluster.obsm["X_latent"][adata_cluster.obs["condition"] == c, :]
            for c in ds_conds
        }
        cluster_latents[dataset][cluster] = cluster_cond_latents


cloudpickle.dump(
    cluster_latents,
    open(
        f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_ref_cluster_latents.pkl", "wb"
    ),
)


#### Filter too small clusters ####
cluster_latents = cloudpickle.load(
    open(
        f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_ref_cluster_latents.pkl", "rb"
    )
)

for dataset in ["neal", "fatima", "nadya"]:
    for cluster in tqdm.tqdm(all_clusters):
        cluster_latent = cluster_latents[dataset][cluster]
        cluster_latent = {c: l for c, l in cluster_latent.items() if l.shape[0] > 30}
        if len(cluster_latent) == 0:
            del cluster_latents[dataset][cluster]
        else:
            cluster_latents[dataset][cluster] = cluster_latent

for cluster in tqdm.tqdm(all_clusters):
    cluster_latent = cluster_latents["observed"][cluster]
    if len(cluster_latent) == 0:
        del cluster_latents["observed"][cluster]
    cluster_latent = {c: l for c, l in cluster_latent.items() if l.shape[0] > 0}
    cluster_latents["observed"][cluster] = cluster_latent


#### Compute distances to reference ####
cluster_dists = []
for dataset in ["neal", "fatima", "nadya", "observed"]:
    ds_latents = cluster_latents[dataset]
    ds_clusters = list(ds_latents.keys())
    for cluster in tqdm.tqdm(ds_clusters):
        cluster_ref = jnp.array(
            adata_ref.obsm["X_scANVI"][adata_ref.obs["Clusters"] == cluster, :]
        )
        cluster_latent = ds_latents[cluster]
        # Sample 300 cells from each cluster
        cluster_latent = {
            c: l[np.random.choice(l.shape[0], 300, replace=True), :]
            for c, l in cluster_latent.items()
        }

        dist_func = lambda x: compute_dists(x, cluster_ref)
        dists = jt.map(dist_func, cluster_latent)
        dists_df = pd.DataFrame(dists).T
        dists_df.index = dists_df.index.rename("condition")
        dists_df.reset_index(inplace=True)
        dists_df["cluster"] = cluster
        dists_df["dataset"] = dataset
        cluster_dists.append(dists_df)

cluster_dist_df = pd.concat(cluster_dists).reset_index(drop=True)
cluster_dist_df.to_csv(
    f"{RESULTS_DIR}/organoid_cond_preds_ref_cluster_dists.tsv",
    sep="\t",
    index=False,
)


#### Save meta ####
adata_pred.obs.to_csv(
    f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_full_meta.tsv", sep="\t"
)

adata_pred.obs.columns

# Summarize to condition & dataset level
# count cells per cluster
cluster_counts = (
    adata_pred.obs.groupby(["dataset", "condition", "Clusters_transfer"])
    .size()
    .reset_index()
)
cluster_counts.columns = ["dataset", "condition", "Clusters_transfer", "n_cells"]

cluster_counts.to_csv(
    f"{RESULTS_DIR}/organoid_cond_preds_cluster_transfer.tsv", sep="\t"
)

region_counts = (
    adata_pred.obs.groupby(["dataset", "condition", "Region_transfer"])
    .size()
    .reset_index()
)
region_counts.columns = ["dataset", "condition", "Region_transfer", "n_cells"]

region_counts.to_csv(f"{RESULTS_DIR}/organoid_cond_preds_region_transfer.tsv", sep="\t")

subregion_counts = (
    adata_pred.obs.groupby(["dataset", "condition", "Subregion_transfer"])
    .size()
    .reset_index()
)
subregion_counts.columns = ["dataset", "condition", "Subregion_transfer", "n_cells"]

subregion_counts.to_csv(
    f"{RESULTS_DIR}/organoid_cond_preds_subregion_transfer.tsv", sep="\t"
)
