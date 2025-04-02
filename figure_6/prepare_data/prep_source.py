import warnings

warnings.simplefilter(action="ignore")

import numpy as np
from matplotlib import pyplot as plt

import anndata as ad
import scanpy as sc
import pandas as pd


#### Source distribution by subs means from pribary atlas (Braun et al.) ####
adata_ref = sc.read_h5ad(
    "/home/fleckj/scratch/data/public_datasets/primary_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v2_common_hv2k_wknn.h5ad"
)
adata = sc.read_h5ad(
    "/home/fleckj/projects/cellflow/data/datasets/AminPasca2023brx/240701_pasca_primary_screen_v1_common_hv2k_wknn.h5ad"
)

ref_latent = adata_ref.obsm["X_scANVI"]
latent_means = []
for _ in range(10000):
    latent_subs = ref_latent[np.random.choice(ref_latent.shape[0], 10, replace=False)]
    latent_means.append(latent_subs.mean(axis=0))

latent_means = np.array(latent_means)

# Dummy adata with zero counts but mean latent
adata_ctrl = ad.AnnData(
    X=np.zeros((10000, adata.n_vars)),
    obs=pd.DataFrame(
        index=[f"CTRL_{i}" for i in range(10000)],
        data={"condition": "CTRL", "morph_comb": "CTRL"},
    ),
    var=adata.var,
    obsm={"X_scanvi_braun": latent_means},
)

adata_comb = ad.concat([adata, adata_ctrl], join="outer")
adata_comb.obs["CTRL"] = adata_comb.obs["condition"] == "CTRL"

p = sc.pl.scatter(
    adata_comb, basis="scanvi_braun", color=["condition", "CTRL"], show=False
)
plt.savefig(
    "/home/fleckj/projects/cellflow/plots/prep_amin/ctrl_ref_dummy_latent.png",
)

adata_comb.write_h5ad(
    "/home/fleckj/projects/cellflow/data/datasets/AminPasca2023brx/240701_pasca_primary_screen_v1_common_hv2k_wknn_mean_ctrl.h5ad"
)
