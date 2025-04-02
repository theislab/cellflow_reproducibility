import jax.tree as jt
from functools import partial
import numpy as np
import scanpy as sc
import pandas as pd

from ott.problems.linear import barycenter_problem
from ott.solvers.linear import continuous_barycenter, sinkhorn

from cfp.metrics import compute_metrics


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

RESULTS_DIR = "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/"

split_tasks = split_meta["split_task"].unique().tolist()


#### Combination splits ####
combination_split_meta = split_meta[split_meta["split_task"] == "combination"]

all_bg_metrics = []
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

        this_split_meta = this_split_meta[~this_split_meta["components"].isna()]
        if this_split_meta.shape[0] == 0:
            continue

        split_metrics = []
        for _, cond in this_split_meta.iterrows():
            gt_data = (
                adata_test[adata_test.obs["condition"] == cond["holdout_condition"]]
                .obsm["X_latent"]
                .copy()
            )
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

            bg_data = {"union": union_latent, "mean": mean_latent, **comp_latent}

            metrics_fun = partial(compute_metrics, gt_data)
            bg_metrics = jt.map(metrics_fun, bg_data)
            bg_metrics_df = pd.DataFrame(bg_metrics).T
            bg_metrics_df["background_condition"] = bg_metrics_df.index
            bg_metrics_df["background_model"] = np.where(
                bg_metrics_df["background_condition"].isin(["mean", "union"]),
                bg_metrics_df["background_condition"],
                "individual",
            )
            bg_metrics_df["holdout_condition"] = cond["holdout_condition"]
            bg_metrics_df["split_name"] = split
            bg_metrics_df["dataset"] = ds
            bg_metrics_df["split_task"] = task
            split_metrics.append(bg_metrics_df)

        split_metrics_df = pd.concat(split_metrics)
        split_metrics_df.to_csv(f"{split_dir}/background_metrics.tsv", sep="\t")

        all_bg_metrics.append(split_metrics_df)

all_bg_metrics_df = pd.concat(all_bg_metrics)
all_bg_metrics_df.to_csv(
    f"{RESULTS_DIR}/combination_background_metrics.tsv",
    sep="\t",
    index=False,
)


#### Transfer splits ####
transfer_split_meta = split_meta[split_meta["split_task"] == "transfer"]

all_bg_metrics = []
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

        adata_train_subs = sc.pp.subsample(adata_train, n_obs=3000, copy=True)
        bg_data = adata_train_subs.obsm["X_latent"]

        split_gt_data = {}
        for _, cond in this_split_meta.iterrows():
            gt_data = (
                adata_test[adata_test.obs["condition"] == cond["holdout_condition"]]
                .obsm["X_latent"]
                .copy()
            )
            split_gt_data[cond["holdout_condition"]] = gt_data

        metrics_fun = lambda x: compute_metrics(x, bg_data)
        split_metrics = jt.map(metrics_fun, split_gt_data)
        split_metrics_df = pd.DataFrame(split_metrics).T
        split_metrics_df["holdout_condition"] = split_metrics_df.index
        split_metrics_df["split_name"] = split
        split_metrics_df["dataset"] = ds
        split_metrics_df["split_task"] = task
        split_metrics_df["background_model"] = "train_dataset"
        split_metrics_df.to_csv(f"{split_dir}/background_metrics.tsv", sep="\t")

        all_bg_metrics.append(split_metrics_df)

all_bg_metrics_df = pd.concat(all_bg_metrics)
all_bg_metrics_df.to_csv(
    f"{RESULTS_DIR}/transfer_background_metrics.tsv",
    sep="\t",
    index=False,
)
