import warnings

warnings.filterwarnings("ignore")

import os

import anndata as ad
import cfp
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from cuml import (
    UMAP,
    LinearRegression,
    LogisticRegression,
)

PLOT_DIR = "/home/fleckj/projects/cellflow/plots/organoid_annots/"
RESULTS_DIR = "/home/fleckj/projects/cellflow/results/organoid_annots/"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/v6/"
FULL_DATA_PATH = f"{DATA_DIR}/basic_split/organoids_combined_full.h5ad"
TEST_DATA_PATH = f"{DATA_DIR}/basic_split/organoids_combined_test.h5ad"
TRAIN_DATA_PATH = f"{DATA_DIR}/basic_split/organoids_combined_train.h5ad"

adata = sc.read(FULL_DATA_PATH)
adata_test = sc.read(TEST_DATA_PATH)
adata_train = sc.read(TRAIN_DATA_PATH)

datasets = np.setdiff1d(adata.obs["dataset"].unique(), "ctrl")
adata_split = {ds: adata[adata.obs["dataset"] == ds].copy() for ds in datasets}
adata_ctrl = adata_train[np.array(adata_train.obs["CTRL"].values), :].copy()

all_combs = np.setdiff1d(adata.obs["mol_comb"].unique(), "CTRL")
all_covar_data = adata.obs[~adata.obs["CTRL"]].iloc[:, :222].drop_duplicates()
all_conds = np.setdiff1d(adata.obs["condition"].unique(), "CTRL")
train_conds = np.setdiff1d(adata_train.obs["condition"].unique(), "CTRL")

test_covar_data = (
    adata_test.obs[~adata_test.obs["CTRL"]].iloc[:, :222].drop_duplicates()
)
test_conds = np.setdiff1d(adata_test.obs["condition"].unique(), "CTRL")
test_combs = np.setdiff1d(adata_test.obs["mol_comb"].unique(), "CTRL")


umap = UMAP(
    n_neighbors=15,
    n_components=2,
    n_epochs=500,
    learning_rate=1.0,
    init="spectral",
    min_dist=0.8,
    spread=1.0,
    negative_sample_rate=5,
    a=None,
    b=None,
    random_state=111,
)

#### Compute umap ####
adata.obsm["X_umap"] = umap.fit_transform(adata.obsm["X_latent"])
cloudpickle.dump(umap, open(f"{RESULTS_DIR}/umap.pkl", "wb"))

p = sc.pl.umap(
    adata,
    color=["leiden_2", "dataset", "subregion_pred_wknn", "class_pred_wknn"],
    show=False,
)
plt.savefig(f"{PLOT_DIR}/umap_all.png", bbox_inches="tight")


#### Annot AP/DV scores ####
adata_ref = sc.read(
    "/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3.1umap_common_hv2k_wknn.h5ad"
)

adata_ref.obs["Subregion"].unique().tolist()

region_ap_values = {
    "Forebrain": 0,
    "Diencephalon": 1,
    "Midbrain": 2,
    "Pons": 3,
    "Cerebellum": 3,
    "Medulla": 4,
}

adata_ref.obs["Region_AP_value"] = adata_ref.obs["Region"].map(region_ap_values)

ap_reg = LinearRegression()

not_nan = ~np.isnan(adata_ref.obs["Region_AP_value"].values)
ap_reg.fit(
    adata_ref.obsm["X_scANVI"][not_nan],
    adata_ref.obs["Region_AP_value"].values[not_nan],
)
adata_ref.obs["Region_AP_value_pred"] = ap_reg.predict(adata_ref.obsm["X_scANVI"])


subregion_ap_values = {
    "Cortex": 0,
    "Striatum": 0.5,
    "Hypothalamus": 1,
    "Thalamus": 2,
    "Midbrain": 3,
    "Pons": 4,
    "Cerebellum": 4,
    "Medulla": 5,
}

adata_ref.obs["Subregion_AP_value"] = adata_ref.obs["Subregion"].map(
    subregion_ap_values
)

not_nan = ~np.isnan(adata_ref.obs["Subregion_AP_value"].values)

ap_reg = LinearRegression()
ap_reg.fit(
    adata_ref.obsm["X_scANVI"][not_nan],
    adata_ref.obs["Subregion_AP_value"].values[not_nan],
)
adata_ref.obs["Subregion_AP_value_pred"] = ap_reg.predict(adata_ref.obsm["X_scANVI"])

adata_ref.obs.groupby("Subregion")["Subregion_AP_value_pred"].mean()

adata_ref.obs["Subregion"].unique().tolist()

subregion_dv_values = {
    "Cortex": 1,
    "Striatum": 0,
    "Hypothalamus": 0,
    "Midbrain ventral": 0,
    "Midbrain dorsal": 1,
    "Pons": 0,
    "Cerebellum": 1,
}

adata_ref.obs["Subregion_DV_value"] = adata_ref.obs["Subregion"].map(
    subregion_dv_values
)

not_nan = ~np.isnan(adata_ref.obs["Subregion_DV_value"].values)

dv_reg = LinearRegression()
dv_reg.fit(
    adata_ref.obsm["X_scANVI"][not_nan],
    adata_ref.obs["Subregion_DV_value"].values[not_nan],
)
adata_ref.obs["Subregion_DV_value_pred"] = dv_reg.predict(adata_ref.obsm["X_scANVI"])


dv_logreg = LogisticRegression()
dv_logreg.fit(
    adata_ref.obsm["X_scANVI"][not_nan],
    adata_ref.obs["Subregion_DV_value"].values[not_nan],
)
adata_ref.obs["Subregion_DV_value_pred_logreg"] = dv_logreg.predict(
    adata_ref.obsm["X_scANVI"]
)


# Ger AP-DV "space"
adata_ref.obsm["X_AP_DV"] = np.stack(
    [
        adata_ref.obs["Region_AP_value_pred"].values,
        adata_ref.obs["Subregion_DV_value_pred"].values,
    ],
    axis=-1,
)

age_reg = LinearRegression()
age_reg.fit(
    adata_ref.obsm["X_scANVI"],
    adata_ref.obs["Age"].values,
)
adata_ref.obs["Age_pred"] = age_reg.predict(adata_ref.obsm["X_scANVI"])

# Save models
cloudpickle.dump(ap_reg, open(f"{RESULTS_DIR}/ap_reg.pkl", "wb"))
cloudpickle.dump(dv_reg, open(f"{RESULTS_DIR}/dv_reg.pkl", "wb"))
cloudpickle.dump(dv_logreg, open(f"{RESULTS_DIR}/dv_logreg.pkl", "wb"))
cloudpickle.dump(age_reg, open(f"{RESULTS_DIR}/age_reg.pkl", "wb"))

adata_ref.write_h5ad(f"{RESULTS_DIR}/braun_2022_fetal_brain_v3.1_annot.h5ad")


#### Predict AP/DV values for GT and predicted data ####
ap_reg = cloudpickle.load(open(f"{RESULTS_DIR}/ap_reg.pkl", "rb"))
dv_reg = cloudpickle.load(open(f"{RESULTS_DIR}/dv_reg.pkl", "rb"))
age_reg = cloudpickle.load(open(f"{RESULTS_DIR}/age_reg.pkl", "rb"))

adata.obs["Region_AP_value_pred_reg"] = ap_reg.predict(adata.obsm["X_latent"])
adata.obs["Subregion_DV_value_pred_logreg"] = dv_reg.predict(adata.obsm["X_latent"])
adata.obs["Age_pred_reg"] = age_reg.predict(adata.obsm["X_latent"])


p = sc.pl.umap(
    adata,
    color=[
        "Region_AP_value_pred_reg",
        "Subregion_DV_value_pred_logreg",
        "Age_pred_reg",
    ],
    show=False,
)
plt.savefig(f"{PLOT_DIR}/umap_annot.png", bbox_inches="tight")

adata.write_h5ad(f"{RESULTS_DIR}/organoids_combined_full_v6_annot.h5ad")


#### Project predicted data ####
TRIAL_NAME = "cellflow_1d81827d"
PRED_DATA_PATH = f"/home/fleckj/projects/cellflow/results/organoid_psweep_v6/basic_split_common/{TRIAL_NAME}/full_gt_pred.h5ad"

adata = sc.read_h5ad(f"{RESULTS_DIR}/organoids_combined_full_v6_annot.h5ad")
adata_gt_pred = sc.read_h5ad(PRED_DATA_PATH)

adata_pred_test = adata_gt_pred[
    np.array((~adata_gt_pred.obs["trained"]) & (adata_gt_pred.obs["split"] == "pred")),
    :,
].copy()

umap = cloudpickle.load(open(f"{RESULTS_DIR}/umap.pkl", "rb"))
adata_pred_test.obsm["X_umap"] = umap.transform(adata_pred_test.obsm["X_latent"])

ap_reg = cloudpickle.load(open(f"{RESULTS_DIR}/ap_reg.pkl", "rb"))
adata_pred_test.obs["Region_AP_value_pred_reg"] = ap_reg.predict(
    adata_pred_test.obsm["X_latent"]
)

dv_reg = cloudpickle.load(open(f"{RESULTS_DIR}/dv_reg.pkl", "rb"))
adata_pred_test.obs["Subregion_DV_value_pred_logreg"] = dv_reg.predict(
    adata_pred_test.obsm["X_latent"]
)

age_reg = cloudpickle.load(open(f"{RESULTS_DIR}/age_reg.pkl", "rb"))
adata_pred_test.obs["Age_pred_reg"] = age_reg.predict(adata_pred_test.obsm["X_latent"])

adata_pred_test.obs["mode"] = "pred"
adata.obs["mode"] = "gt"

cfp.pp.compute_wknn(
    adata_ref,
    adata_pred_test,
    ref_rep_key="X_scANVI",
    query_rep_key="X_latent",
)
for label_key in ["Subregion", "Region", "CellClass", "Age"]:
    cfp.pp.transfer_labels(
        adata_pred_test,
        adata_ref,
        label_key=label_key,
    )
    adata_pred_test.obs[f"{label_key.lower()}_pred_wknn"] = adata_pred_test.obs[
        f"{label_key}_transfer"
    ]
    adata_pred_test.obs.drop(f"{label_key}_transfer", axis=1, inplace=True)
    adata_pred_test.obs.drop(f"{label_key}_transfer_score", axis=1, inplace=True)

adata_gt_pred = ad.concat([adata, adata_pred_test], join="outer")

p = sc.pl.umap(
    adata_gt_pred,
    color=[
        "mode",
        "subregion_pred_wknn",
    ],
    show=False,
)
plt.savefig(f"{PLOT_DIR}/umap_pred_annot.png", bbox_inches="tight")

adata_pred_test.write_h5ad(f"{RESULTS_DIR}/organoids_combined_pred_v6_annot.h5ad")
