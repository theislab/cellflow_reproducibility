import jax.tree as jt
from functools import partial
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

from ott.problems.linear import barycenter_problem
from ott.solvers.linear import continuous_barycenter, sinkhorn

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance, entropy

import cfp
from cfp.metrics import compute_metrics


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


def barycenter_interpolation(*args, epsilon=0.1, **kwargs):
    joint_data = np.concatenate(args, axis=0)
    num_per_segment = [data.shape[0] for data in args]
    bar_p = barycenter_problem.FreeBarycenterProblem(
        y=joint_data,
        epsilon=epsilon,
        num_per_segment=num_per_segment,
    )
    linear_solver = sinkhorn.Sinkhorn(lse_mode=True, **kwargs)
    solver = continuous_barycenter.FreeWassersteinBarycenter(linear_solver)
    out = solver(bar_p, bar_size=joint_data.shape[0])
    return out.x


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


#### Combination splits ####
combination_split_meta = split_meta[split_meta["split_task"] == "combination"]

all_dist_metrics = []
all_cluster_metrics = []
datasets = combination_split_meta["dataset"].unique().tolist()
for ds in datasets:
    print(ds)

    ds_meta = combination_split_meta[combination_split_meta["dataset"] == ds]
    splits = ds_meta["split_name"].unique().tolist()

    for split in splits:
        print(split)

        this_split_meta = ds_meta[ds_meta["split_name"] == split]
        holdout_conds = this_split_meta["split_name"].tolist()

        task = "combination"
        split_dir = f"{RESULTS_DIR}/{task}/{ds}/{split}/"

        adata_train = sc.read_h5ad(f"{split_dir}/adata_train.h5ad")
        adata_test = sc.read_h5ad(f"{split_dir}/adata_test.h5ad")
        adata_full = sc.read_h5ad(f"{split_dir}/adata_full.h5ad")

        this_split_meta = this_split_meta[~this_split_meta["components"].isna()]
        if this_split_meta.shape[0] == 0:
            continue

        union_baselines = []
        mean_baselines = []
        for _, cond in this_split_meta.iterrows():
            components = cond["components"].split(";")
            comp_latent = {
                comp: adata_train.obsm["X_latent"][
                    adata_train.obs["condition"] == comp, :
                ]
                for comp in components
            }
            comp_latent_sizes = [latent.shape[0] for latent in comp_latent.values()]
            min_size = np.min(comp_latent_sizes)
            # Subsample to smallest size
            for comp in comp_latent:
                comp_latent[comp] = comp_latent[comp][
                    np.random.choice(min_size, min_size)
                ]
            union_latent = np.concatenate(list(comp_latent.values()), axis=0)
            mean_latent = barycenter_interpolation(
                *list(comp_latent.values()), epsilon=0.1
            )

            union_adata = sc.AnnData(
                X=np.zeros((union_latent.shape[0], 1)),
                obs={"condition": cond["holdout_condition"]},
                obsm={"X_latent": union_latent},
            )

            mean_adata = sc.AnnData(
                X=np.zeros((mean_latent.shape[0], 1)),
                obs={"condition": cond["holdout_condition"]},
                obsm={"X_latent": mean_latent},
            )

            union_baselines.append(union_adata)
            mean_baselines.append(mean_adata)

        union_adata = ad.concat(union_baselines, join="outer")
        mean_adata = ad.concat(mean_baselines, join="outer")

        adata_baselines = ad.concat(
            {"union": union_adata, "mean": mean_adata}, join="outer", label="model"
        )

        cfp.pp.compute_wknn(
            adata_full,
            adata_baselines,
            ref_rep_key="X_latent",
            query_rep_key="X_latent",
        )

        for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
            cfp.pp.transfer_labels(
                adata_baselines,
                adata_full,
                label_key=label_key,
            )

        adata_baselines.write_h5ad(f"{split_dir}/adata_baselines.h5ad")

        union_adata = adata_baselines[adata_baselines.obs["model"] == "union"]
        mean_adata = adata_baselines[adata_baselines.obs["model"] == "mean"]
        holdout_conds = this_split_meta["holdout_condition"].tolist()

        gt_data = {
            cond: adata_test[adata_test.obs["condition"] == cond]
            .obsm["X_latent"]
            .copy()
            for cond in holdout_conds
        }

        union_data = {
            cond: union_adata[union_adata.obs["condition"] == cond]
            .obsm["X_latent"]
            .copy()
            for cond in holdout_conds
        }

        mean_data = {
            cond: mean_adata[mean_adata.obs["condition"] == cond]
            .obsm["X_latent"]
            .copy()
            for cond in holdout_conds
        }

        union_metrics = jt.map(compute_metrics, gt_data, union_data)
        union_metrics_df = pd.DataFrame(union_metrics).T
        union_metrics_df["holdout_condition"] = union_metrics_df.index
        union_metrics_df["model"] = "union"

        mean_metrics = jt.map(compute_metrics, gt_data, mean_data)
        mean_metrics_df = pd.DataFrame(mean_metrics).T
        mean_metrics_df["holdout_condition"] = mean_metrics_df.index
        mean_metrics_df["model"] = "barycenter"

        dist_metrics_df = pd.concat([union_metrics_df, mean_metrics_df])
        dist_metrics_df["split_name"] = split
        dist_metrics_df["dataset"] = ds
        dist_metrics_df["split_task"] = task

        dist_metrics_df.to_csv(f"{split_dir}/baseline_dist_metrics.tsv", sep="\t")

        cluster_metrics_results = {}
        for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
            gt_props = {
                cond: adata_full[adata_full.obs["condition"] == cond]
                .obs[label_key]
                .value_counts(normalize=True)
                for cond in holdout_conds
            }

            union_props = {
                cond: union_adata[union_adata.obs["condition"] == cond]
                .obs[f"{label_key}_transfer"]
                .value_counts(normalize=True)
                .reindex(gt_props[cond].index)
                .fillna(0)
                for cond in holdout_conds
            }

            mean_props = {
                cond: mean_adata[mean_adata.obs["condition"] == cond]
                .obs[f"{label_key}_transfer"]
                .value_counts(normalize=True)
                .reindex(gt_props[cond].index)
                .fillna(0)
                for cond in holdout_conds
            }

            union_cluster_metrics = jt.map(
                compute_cluster_metrics, gt_props, union_props
            )
            union_cluster_metrics_df = pd.DataFrame(union_cluster_metrics).T
            union_cluster_metrics_df["holdout_condition"] = (
                union_cluster_metrics_df.index
            )
            union_cluster_metrics_df["model"] = "union"

            mean_cluster_metrics = jt.map(compute_cluster_metrics, gt_props, mean_props)
            mean_cluster_metrics_df = pd.DataFrame(mean_cluster_metrics).T
            mean_cluster_metrics_df["holdout_condition"] = mean_cluster_metrics_df.index
            mean_cluster_metrics_df["model"] = "barycenter"

            cluster_metrics_df = pd.concat(
                [union_cluster_metrics_df, mean_cluster_metrics_df]
            )
            cluster_metrics_df["split_name"] = split
            cluster_metrics_df["dataset"] = ds
            cluster_metrics_df["split_task"] = task
            cluster_metrics_df["cluster_key"] = label_key
            cluster_metrics_results[label_key] = cluster_metrics_df

        cluster_metrics_df = pd.concat(cluster_metrics_results.values())
        cluster_metrics_df.to_csv(f"{split_dir}/baseline_cluster_metrics.tsv", sep="\t")

        all_cluster_metrics.append(cluster_metrics_df)
        all_dist_metrics.append(dist_metrics_df)


all_cluster_metrics_df = pd.concat(all_cluster_metrics)
all_cluster_metrics_df.to_csv(
    f"{RESULTS_DIR}/combination_baseline_cluster_metrics.tsv",
    sep="\t",
    index=False,
)

all_dist_metrics_df = pd.concat(all_dist_metrics)
all_dist_metrics_df.to_csv(
    f"{RESULTS_DIR}/combination_baseline_dist_metrics.tsv",
    sep="\t",
    index=False,
)


#### Transfer splits ####
transfer_split_meta = split_meta[split_meta["split_task"] == "transfer"]

all_cluster_metrics = []
all_dist_metrics = []
datasets = transfer_split_meta["dataset"].unique().tolist()
for ds in datasets:
    print(ds)

    ds_meta = transfer_split_meta[transfer_split_meta["dataset"] == ds]
    splits = ds_meta["split_name"].unique().tolist()

    for split in splits:
        print(split)

        this_split_meta = ds_meta[ds_meta["split_name"] == split]
        holdout_conds = this_split_meta["split_name"].tolist()

        task = "transfer"
        split_dir = f"{RESULTS_DIR}/{task}/{ds}/{split}/"

        adata_train = sc.read_h5ad(f"{split_dir}/adata_train.h5ad")
        adata_test = sc.read_h5ad(f"{split_dir}/adata_test.h5ad")
        adata_full = sc.read_h5ad(f"{split_dir}/adata_full.h5ad")

        adata_ds_train = adata_full[
            (adata_full.obs["dataset"] == ds) & (adata_full.obs["split"] == "train")
        ]

        adata_train_subs = sc.pp.subsample(adata_ds_train, n_obs=3000, copy=True)
        bg_data = adata_train_subs.obsm["X_latent"]

        holdout_conds = this_split_meta["holdout_condition"].tolist()
        gt_data = {
            cond: adata_test[adata_test.obs["condition"] == cond]
            .obsm["X_latent"]
            .copy()
            for cond in holdout_conds
        }

        metrics_fun = lambda x: compute_metrics(x, bg_data)
        dist_metrics = jt.map(metrics_fun, gt_data)
        dist_metrics_df = pd.DataFrame(dist_metrics).T
        dist_metrics_df["holdout_condition"] = dist_metrics_df.index
        dist_metrics_df["split_name"] = split
        dist_metrics_df["dataset"] = ds
        dist_metrics_df["split_task"] = task
        dist_metrics_df["model"] = "train_dataset"
        dist_metrics_df.to_csv(f"{split_dir}/baseline_dist_metrics.tsv", sep="\t")

        cluster_metrics_results = {}
        for label_key in ["leiden_2", "leiden_3", "leiden_4"]:
            gt_props = {
                cond: adata_full[adata_full.obs["condition"] == cond]
                .obs[label_key]
                .value_counts(normalize=True)
                for cond in holdout_conds
            }

            bg_props = {
                cond: adata_train_subs.obs[label_key]
                .value_counts(normalize=True)
                .reindex(gt_props[cond].index)
                .fillna(0)
                for cond in holdout_conds
            }

            bg_metrics = jt.map(compute_cluster_metrics, gt_props, bg_props)
            bg_metrics_df = pd.DataFrame(bg_metrics).T
            bg_metrics_df["holdout_condition"] = bg_metrics_df.index
            bg_metrics_df["model"] = "train_dataset"
            bg_metrics_df["cluster_key"] = label_key
            bg_metrics_df["split_name"] = split
            bg_metrics_df["dataset"] = ds
            bg_metrics_df["split_task"] = task
            cluster_metrics_results[label_key] = bg_metrics_df

        cluster_metrics_df = pd.concat(cluster_metrics_results.values())
        cluster_metrics_df.to_csv(f"{split_dir}/baseline_cluster_metrics.tsv", sep="\t")

        all_cluster_metrics.append(cluster_metrics_df)
        all_dist_metrics.append(dist_metrics_df)


all_cluster_metrics_df = pd.concat(all_cluster_metrics)
all_cluster_metrics_df.to_csv(
    f"{RESULTS_DIR}/transfer_baseline_cluster_metrics.tsv",
    sep="\t",
    index=False,
)

all_dist_metrics_df = pd.concat(all_dist_metrics)
all_dist_metrics_df.to_csv(
    f"{RESULTS_DIR}/transfer_baseline_dist_metrics.tsv",
    sep="\t",
    index=False,
)
