import warnings

warnings.filterwarnings("ignore")

import os

import jax.tree as jt
import jax.numpy as jnp
from functools import partial
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from plotnine import *
from cuml import UMAP
import argparse

import cfp


#### Parse arguments ####
parser = argparse.ArgumentParser()
parser.add_argument("--full_adata", type=str, help="Path to full data")
parser.add_argument("--name", type=str, help="Name of the run")
parser.add_argument("--trial_name", type=str, help="Name of the ray trial")
args = parser.parse_args()


#### Project name & dir ####
PROJECT_NAME = "train_eval_organoids_common"
RESULTS_DIR = f"/home/fleckj/projects/cellflow/results/"
OUT_DIR = os.path.join(RESULTS_DIR, PROJECT_NAME, args.trial_name, args.name)


#### Load predicted data ####
adata_full = sc.read_h5ad(args.full_adata)
adata_pred = sc.read_h5ad(f"{OUT_DIR}/test_predictions.h5ad")
test_conds = np.setdiff1d(adata_pred.obs["condition"].unique(), "CTRL")
conditions = adata_pred.obs["condition"].unique().tolist()


#### Transfer cluster labels ####
cfp.pp.compute_wknn(
    adata_full,
    adata_pred,
    ref_rep_key="X_latent",
    query_rep_key="X_latent",
)

for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
    cfp.pp.transfer_labels(
        adata_pred,
        adata_full,
        label_key=label_key,
    )


#### Compute umap projection ####
umap = UMAP(
    n_neighbors=15,
    n_components=2,
    n_epochs=500,
    learning_rate=1.0,
    init="spectral",
    min_dist=0.3,
    spread=1.0,
    negative_sample_rate=5,
    a=None,
    b=None,
    random_state=111,
)

#### Compute projected umap and join ####
umap.fit(adata_full.obsm["X_latent"])
adata_full.obsm["X_umap"] = umap.transform(adata_full.obsm["X_latent"])
adata_pred.obsm["X_umap"] = umap.transform(adata_pred.obsm["X_latent"])
adata_pred.obsm["X_umap"] = np.clip(
    adata_pred.obsm["X_umap"],
    adata_full.obsm["X_umap"].min(),
    adata_full.obsm["X_umap"].max(),
)

adata_gt_pred = ad.concat(
    {"pred": adata_pred, "gt": adata_full}, join="outer", label="split"
)

adata_gt_pred.write_h5ad(f"{OUT_DIR}/gt_test_predictions.h5ad")


#### Compute cluster precision and recall ####
def detected_clusters(x, threshold=0.05):
    value_counts = jnp.unique_counts(x)
    value_frac = value_counts[1] / x.shape[0]
    select_vals = value_counts[0][value_frac > threshold]
    return select_vals


def precision_recall(x, y, x_threshold=0.05, y_thresholds=[0.05]):
    precisions = []
    recalls = []
    for threshold in y_thresholds:
        x_detect = detected_clusters(x, threshold=x_threshold)
        y_detect = detected_clusters(y, threshold=threshold)
        # Get true positives, false positives, and false negatives
        tp = jnp.intersect1d(x_detect, y_detect).shape[0]
        fp = jnp.setdiff1d(y_detect, x_detect).shape[0]
        fn = jnp.setdiff1d(x_detect, y_detect).shape[0]
        # Compute precision and recall
        # But catch division by zero
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    return {"precision": precisions, "recall": recalls}


thresholds = np.arange(1, 20, 0.5) / 100
precrec_results = {}
for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
    pred_cluster_data = {
        cond: jnp.array(
            adata_pred.obs[f"{label_key}_transfer"][
                adata_pred.obs["condition"] == cond
            ].values.astype(int)
        )
        for cond in test_conds
    }

    gt_cluster_data = {
        cond: jnp.array(
            adata_full.obs[label_key][
                adata_full.obs["condition"] == cond
            ].values.astype(int)
        )
        for cond in test_conds
    }
    precrec_func = partial(precision_recall, y_thresholds=thresholds)
    precrec = jt.map(precrec_func, gt_cluster_data, pred_cluster_data)
    precrec_df = pd.concat(
        {cond: pd.DataFrame(precrec[cond]) for cond in precrec}, names=["condition"]
    ).reset_index()
    precrec_df["threshold"] = np.concatenate([thresholds for _ in test_conds], axis=0)
    precrec_df["cluster_key"] = label_key
    precrec_df = precrec_df.drop(columns="level_1")
    precrec_results[label_key] = precrec_df

precrec_df = pd.concat(precrec_results.values())
precrec_df.to_csv(f"{OUT_DIR}/precrec_metrics.tsv", sep="\t")


#### Compute other cluster metrics ####
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance, entropy


def compute_cluster_metrics(
    true_props: np.ndarray, pred_props: np.ndarray
) -> dict[str, float]:
    metrics = {
        "cosine": cosine(true_props, pred_props),
        "pcorr": pearsonr(true_props, pred_props)[0],
        "wasserstein": wasserstein_distance(true_props, pred_props),
        "kd_truefirst": entropy(true_props, pred_props),
        "kd_predfirst": entropy(pred_props, true_props),
        "mae": np.mean(np.abs(true_props - pred_props)),
    }
    return metrics


cluster_metrics_results = {}
for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
    pred_cluster_data = {
        cond: jnp.array(
            adata_pred.obs[f"{label_key}_transfer"][
                adata_pred.obs["condition"] == cond
            ].values.astype(int)
        )
        for cond in test_conds
    }
    true_props = {
        cond: adata_full[adata_full.obs["condition"] == cond]
        .obs[label_key]
        .value_counts(normalize=True)
        for cond in conditions
    }
    pred_props = {
        cond: adata_pred[adata_pred.obs["condition"] == cond]
        .obs[f"{label_key}_transfer"]
        .value_counts(normalize=True)
        .reindex(true_props[cond].index)
        .fillna(0)
        for cond in conditions
    }
    cluster_metrics = jt.map(compute_cluster_metrics, true_props, pred_props)
    cluster_metrics_df = pd.DataFrame(cluster_metrics).T
    cluster_metrics_df["cluster_key"] = label_key
    cluster_metrics_results[label_key] = cluster_metrics_df

cluster_metrics_df = pd.concat(cluster_metrics_results.values())
cluster_metrics_df.to_csv(f"{OUT_DIR}/cluster_metrics.tsv", sep="\t")
