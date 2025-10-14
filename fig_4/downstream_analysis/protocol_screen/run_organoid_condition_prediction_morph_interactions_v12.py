import warnings

warnings.filterwarnings("ignore")

import itertools
import os

import cloudpickle
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from cellflow.metrics import compute_e_distance_fast, compute_scalar_mmd

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


#### Prepare distributions ####
timing_cols = cond_meta.loc[:, cond_meta.columns.str.contains("timing")]
all_comb_mols = timing_cols.columns.str.replace("_timing", "").unique().tolist()
combs1 = [(mol,) for mol in all_comb_mols]
combs2 = list(itertools.combinations(all_comb_mols, 2))
combs3 = list(itertools.combinations(all_comb_mols, 3))
timings = ["early", "mid", "late", "early-late", "early-mid", "mid-late"]

all_combs = [*combs1, *combs2, *combs3]

#### Get unions and combinations of morphogen conditions ####
pred_single = {}
pred_combs = {}
pred_unions = {}
comb_meta = []
np.random.seed(111)
for dataset in ["neal", "fatima", "nadya"]:
    pred_combs[dataset] = {}
    pred_unions[dataset] = {}
    pred_single[dataset] = {}

    ds_conds = cond_meta[cond_meta["dataset"] == dataset].reset_index(drop=True)
    ds_adata = adata_pred[adata_pred.obs["dataset"] == dataset, :]

    for comb in tqdm.tqdm(all_combs):
        comb_str = "+".join(comb)

        if len(comb) == 1:
            pred_single[dataset][comb_str] = {}
        else:
            pred_combs[dataset][comb_str] = {}
            pred_unions[dataset][comb_str] = {}

        comb_cols = [f"{mol}_timing" for mol in comb]
        # Conditions with both morphogens present
        has_all = (~ds_conds[comb_cols].isna()).all(1)
        has_comb = (~ds_conds[timing_cols.columns].isna()).sum(1) == len(comb)
        has_comb = has_comb & (ds_conds["dataset"] == dataset)
        comb_conds = ds_conds.loc[has_comb & has_all,].copy().reset_index(drop=True)

        for _, cond in comb_conds.iterrows():
            time_str = "+".join(cond[comb_cols].values.tolist())

            cond["time_comb"] = time_str
            cond["comb"] = comb_str
            comb_meta.append(cond)

            if len(comb) == 1:
                single_latent = (
                    ds_adata[ds_adata.obs["condition"] == cond["condition"], :]
                    .obsm["X_latent"]
                    .copy()
                )

                pred_single[dataset][comb_str][time_str] = single_latent[
                    np.random.choice(single_latent.shape[0], 3000, replace=False), :
                ]

                continue

            # Conditions with exactly one morphogen present
            individual_conds = ds_conds.loc[
                ((ds_conds[comb_cols] == cond[comb_cols]).sum(1) == 1)
                & (ds_conds["n_mols"] == 1),
                :,
            ]

            pred_comb_latent = (
                ds_adata[ds_adata.obs["condition"] == cond["condition"], :]
                .obsm["X_latent"]
                .copy()
            )
            pred_combs[dataset][comb_str][time_str] = pred_comb_latent[
                np.random.choice(pred_comb_latent.shape[0], 3000, replace=False), :
            ]

            pred_union_latent = (
                ds_adata[
                    ds_adata.obs["condition"].isin(individual_conds["condition"]), :
                ]
                .obsm["X_latent"]
                .copy()
            )
            pred_unions[dataset][comb_str][time_str] = pred_union_latent[
                np.random.choice(pred_union_latent.shape[0], 6000, replace=False), :
            ]

    cloudpickle.dump(
        pred_single[dataset],
        open(
            f"{RESULTS_DIR}/morphogen_interaction_preds_single_{dataset}.pkl",
            "wb",
        ),
    )

    cloudpickle.dump(
        pred_combs[dataset],
        open(
            f"{RESULTS_DIR}/morphogen_interaction_preds_comb_{dataset}.pkl",
            "wb",
        ),
    )

    cloudpickle.dump(
        pred_unions[dataset],
        open(
            f"{RESULTS_DIR}/morphogen_interaction_preds_union_{dataset}.pkl",
            "wb",
        ),
    )

comb_meta_full = pd.DataFrame(comb_meta)
comb_meta_full["comb_sorted"] = comb_meta_full["comb"].apply(
    lambda x: "+".join(sorted(x.split("+")))
)
comb_meta_full.to_csv(
    f"{RESULTS_DIR}/morphogen_interaction_meta.tsv", sep="\t", index=False
)


#### Compute MMD of comb vs single ####
interaction_mmd = jt.map(compute_scalar_mmd, pred_combs, pred_unions)
interaction_mmd_df = pd.concat(
    {k: pd.DataFrame(v).T for k, v in interaction_mmd.items()}, axis=0
)
interaction_mmd_df.index = interaction_mmd_df.index.rename(["dataset", "comb"])
interaction_mmd_df.reset_index(inplace=True)
interaction_mmd_df_long = interaction_mmd_df.melt(
    id_vars=["dataset", "comb"], var_name="time_comb", value_name="mmd"
)
interaction_mmd_df_long = interaction_mmd_df_long[
    ~interaction_mmd_df_long["mmd"].isna()
]

interaction_mmd_df_long.to_csv(
    f"{RESULTS_DIR}/morphogen_interaction_mmd.tsv", sep="\t", index=False
)


#### Compute edistance between timepoints for each condition and dataset ####
## Get condition representations for all condition with constant time points ####
pred_conds = {}
np.random.seed(111)
for dataset in ["neal", "fatima", "nadya"]:
    pred_conds[dataset] = {}

    ds_conds = cond_meta[cond_meta["dataset"] == dataset].reset_index(drop=True)
    ds_adata = adata_pred[adata_pred.obs["dataset"] == dataset, :]

    for comb in tqdm.tqdm(all_combs):
        comb_str = "+".join(comb)

        pred_conds[dataset][comb_str] = {}

        comb_cols = [f"{mol}_timing" for mol in comb]
        # Conditions with simulataneous morphogen timings
        has_all = (~ds_conds[comb_cols].isna()).all(1) & (
            ds_conds["dataset"] == dataset
        )
        same_time = (ds_conds[comb_cols].nunique(1) == 1) & (
            ds_conds["n_mols"] == len(comb)
        )
        comb_conds = ds_conds.loc[same_time & has_all,].copy().reset_index(drop=True)

        for _, cond in comb_conds.iterrows():
            time_str = cond[comb_cols].unique().item()

            cond["time_comb"] = time_str
            cond["comb"] = comb_str
            comb_meta.append(cond)

            # Get condition representation
            cond_latent = (
                ds_adata[ds_adata.obs["condition"] == cond["condition"], :]
                .obsm["X_latent"]
                .copy()
            )
            pred_conds[dataset][comb_str][time_str] = cond_latent[
                np.random.choice(cond_latent.shape[0], 3000, replace=False), :
            ]

    cloudpickle.dump(
        pred_conds[dataset],
        open(
            f"{RESULTS_DIR}/morphogen_timepoint_preds_{dataset}.pkl",
            "wb",
        ),
    )


### Compute e-dist between timepoints for each condition and dataset ####
all_datasets = ["neal", "fatima", "nadya"]
timing_edist_list = []
for dataset in all_datasets:
    print(dataset)
    dataset_reps = pred_conds[dataset]

    for comb, timepoints in tqdm.tqdm(dataset_reps.items()):
        for timing, time_rep in timepoints.items():
            timing_edist = jt.map(
                lambda x: compute_e_distance_fast(x, time_rep), timepoints
            )
            timing_edist_df = pd.DataFrame(timing_edist, index=timepoints.keys()).melt(
                var_name="timepoint_2", value_name="e_distance"
            )
            timing_edist_df["timepoint_1"] = timing
            timing_edist_df["comb"] = comb
            timing_edist_df["dataset"] = dataset
            timing_edist_list.append(timing_edist_df)

timing_edist_df = pd.concat(timing_edist_list).reset_index(drop=True)
timing_edist_df = timing_edist_df[
    timing_edist_df["timepoint_1"] != timing_edist_df["timepoint_2"]
]
timing_edist_df.to_csv(
    f"{RESULTS_DIR}/morphogen_timepoint_e_distance.tsv", sep="\t", index=False
)


#### Load pred combs ####
pred_combs = {
    k: cloudpickle.load(
        open(f"{RESULTS_DIR}/morphogen_interaction_preds_comb_{k}.pkl", "rb")
    )
    for k in ["neal", "fatima", "nadya"]
}


#### Compute MMD of comb vs full dataset ####
full_latent = adata.obsm["X_latent"][
    np.random.choice(adata.obsm["X_latent"].shape[0], 30000, replace=False), :
]

dataset_mmd = jt.map(lambda x: compute_scalar_mmd(x, full_latent), pred_combs)

dataset_mmd_df = pd.concat(
    {k: pd.DataFrame(v).T for k, v in dataset_mmd.items()}, axis=0
)
dataset_mmd_df.index = dataset_mmd_df.index.rename(["dataset", "comb"])
dataset_mmd_df.reset_index(inplace=True)
dataset_mmd_df_long = dataset_mmd_df.melt(
    id_vars=["dataset", "comb"], var_name="time_comb", value_name="mmd"
)
dataset_mmd_df_long = dataset_mmd_df_long[~dataset_mmd_df_long["mmd"].isna()]

dataset_mmd_df_long.to_csv(
    f"{RESULTS_DIR}/morphogen_dataset_mmd.tsv", sep="\t", index=False
)


#### Compute e distance vs individual datasets ####
ds_edist_list = []
for dataset in ["neal", "fatima", "nadya"]:
    print(dataset)

    ds_latent = adata[adata.obs["dataset"] == dataset, :].obsm["X_latent"]
    ds_latent = jnp.array(ds_latent)
    ds_latent = ds_latent[np.random.choice(ds_latent.shape[0], 30000, replace=True), :]

    ds_edist = jt.map(
        lambda x: compute_e_distance_fast(x, ds_latent), pred_combs[dataset]
    )
    ds_edist_df = pd.DataFrame(ds_edist)
    ds_edist_df.index = ds_edist_df.index.rename("time_comb")
    ds_edist_df.reset_index(inplace=True)
    ds_edist_df_long = ds_edist_df.melt(
        id_vars=["time_comb"], var_name="comb", value_name="e_distance"
    )
    ds_edist_df_long["dataset"] = dataset
    ds_edist_list.append(ds_edist_df_long)

ds_edist_df = pd.concat(ds_edist_list).reset_index(drop=True)
ds_edist_df.to_csv(
    f"{RESULTS_DIR}/morphogen_dataset_e_distance.tsv", sep="\t", index=False
)


#### Compute MMD of comb vs all observed conditions ####
all_conditions = np.setdiff1d(adata.obs["condition"].unique().tolist(), "CTRL")

condition_mmd = {}
for cond in tqdm.tqdm(all_conditions):
    print(cond)

    cond_latent = adata_pred[adata_pred.obs["condition"] == cond, :].obsm["X_latent"]
    cond_latent = jnp.array(cond_latent)
    cond_latent = cond_latent[
        np.random.choice(cond_latent.shape[0], 1000, replace=True), :
    ]

    cond_mmd = jt.map(lambda x: compute_scalar_mmd(x, cond_latent), pred_combs)
    cond_mmd_df = pd.concat({k: pd.DataFrame(v).T for k, v in cond_mmd.items()}, axis=0)
    cond_mmd_df.index = cond_mmd_df.index.rename(["dataset", "comb"])
    cond_mmd_df.reset_index(inplace=True)
    cond_mmd_df["condition"] = cond
    cond_mmd_df_long = cond_mmd_df.melt(
        id_vars=["dataset", "comb", "condition"], var_name="time_comb", value_name="mmd"
    )
    cond_mmd_df_long = cond_mmd_df_long[~cond_mmd_df_long["mmd"].isna()]
    condition_mmd[cond] = cond_mmd_df_long


condition_mmd_df = pd.concat(condition_mmd).reset_index(drop=True)
condition_mmd_df.to_csv(
    f"{RESULTS_DIR}/morphogen_condition_mmd.tsv", sep="\t", index=False
)
