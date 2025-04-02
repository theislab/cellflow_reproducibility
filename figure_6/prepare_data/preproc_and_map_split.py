import os
import sys
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

from hnoca.map import AtlasMapper
from hnoca.map.utils import prepare_features


SPLIT_DIR = sys.argv[1]


#### Load split data ####
RESULTS_DIR = "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/"

combined_count_adata = sc.read_h5ad(f"{RESULTS_DIR}/org_combined_counts.h5ad")

split_meta = pd.read_csv(
    f"{RESULTS_DIR}/organoid_holdout_splits.tsv",
    sep="\t",
)

adata_train = ad.read(f"{SPLIT_DIR}/adata_train.h5ad")
adata_test = ad.read(f"{SPLIT_DIR}/adata_test.h5ad")

X_latent_ctrl = adata_train.obsm["X_latent"][adata_train.obs["CTRL"]]
X_latent_ctrl_test = adata_test.obsm["X_latent"][adata_test.obs["CTRL"]]


#### Load atlas mapper ####
REF_PATH = "/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3cleaned_common_hv2k_wknn.h5ad"
MODEL_DIR = "/home/fleckj/projects/atlas/data/results/atlasv3_scanvi_scarches/v7/subregion_class/query_scarches_nolabs"

adata_ref = sc.read(REF_PATH)
adata_ref.varm = dict()
adata_ref.obs["batch"] = adata_ref.obs["Donor"].astype(str).copy()

ref_vae = scvi.model.SCANVI.load(
    os.path.join(MODEL_DIR, "model.pt"),
    adata=adata_ref,
)

#### Project train data with partial retraining ####
adata_q = combined_count_adata[adata_train[~adata_train.obs["CTRL"]].obs_names].copy()
adata_q.obs["batch"] = adata_q.obs["dataset"].copy()
adata_q.obs["subregion_class"] = "Unknown"
adata_q.obs["subregion_class"] = adata_q.obs["subregion_class"].astype("category")

# Finetune on the query data
mapper = AtlasMapper(ref_vae)
mapper.map_query(
    adata_q,
    retrain="partial",
    max_epochs=200,
    batch_size=1024,
    plan_kwargs=dict(weight_decay=0.0),
)

mapper.save(f"{SPLIT_DIR}/mapper/")

# Get latent for train data
mapper = AtlasMapper.load(f"{SPLIT_DIR}/mapper/")

X_latent_train = mapper.get_latent_representation()
adata_train.obsm["X_latent"] = np.concatenate([X_latent_train, X_latent_ctrl])

# Get latent for test data without retraining
adata_q_test = prepare_features(
    combined_count_adata[adata_test.obs_names[~adata_test.obs["CTRL"]]],
    ref_vae,
)[:, ref_vae.adata.var_names].copy()
adata_q_test.obs["batch"] = adata_q_test.obs["dataset"].copy()
adata_q_test.obs["subregion_class"] = "Unknown"
adata_q_test.obs["subregion_class"] = adata_q_test.obs["subregion_class"].astype(
    "category"
)

X_latent_test = mapper.get_latent_representation(adata_q_test)
adata_test.obsm["X_latent"] = np.concatenate([X_latent_test, X_latent_ctrl_test])

#### Join to get full dataset and cluster ####
adata_full = ad.concat(
    {
        "train": adata_train,
        "test": adata_test,
    },
    label="split",
    join="outer",
)

adata_full = adata_full[~adata_full.obs["CTRL"]].copy()

# Cluster at three resolutions
rsc.pp.neighbors(adata_full, use_rep="X_latent", n_neighbors=30)
rsc.tl.leiden(adata_full, resolution=2.0)
adata_full.obs["leiden_2"] = adata_full.obs["leiden"].copy()
rsc.tl.leiden(adata_full, resolution=3.0)
adata_full.obs["leiden_3"] = adata_full.obs["leiden"].copy()
rsc.tl.leiden(adata_full, resolution=4.0)
adata_full.obs["leiden_4"] = adata_full.obs["leiden"].copy()

# UMAP
adata_full_plot = sc.pp.subsample(adata_full, n_obs=100000, copy=True)
rsc.pp.neighbors(adata_full_plot, use_rep="X_latent", n_neighbors=30)
rsc.tl.umap(adata_full_plot)

p = sc.pl.umap(
    adata_full_plot,
    color=["dataset", "split", "leiden", "region_pred_wknn"],
    ncols=2,
    show=False,
)

os.makedirs(f"{SPLIT_DIR}/plots", exist_ok=True)
plt.savefig(f"{SPLIT_DIR}/plots/umap.png", dpi=300)

adata_full.write_h5ad(f"{SPLIT_DIR}/adata_full.h5ad")
adata_train.write_h5ad(f"{SPLIT_DIR}/adata_train.h5ad")
adata_test.write_h5ad(f"{SPLIT_DIR}/adata_test.h5ad")
