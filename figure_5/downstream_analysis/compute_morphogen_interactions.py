import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import warnings
import itertools
from cfp.metrics import compute_scalar_mmd

warnings.filterwarnings("ignore")

####################
## Configurations ##
####################

# whether to compute MMD for triples from three single conditions
# if False, compute MMD from AP double condition and DV condition
TRIPLE_FROM_THREE_SINGLE = False

# files and directories
GT_ADATA = "/data/bobrdan/thesis/data/ineurons_glutpost.h5ad"
CF_DIR = "/data/bobrdan/thesis/results/bestsw/old/20241130_101236_868/"
OUTPUT_DIR = "/data/bobrdan/thesis/"


###############
## Load data ##
###############

adatas = []
for subdir in os.listdir(CF_DIR):
    subdir = os.path.join(CF_DIR, subdir)
    for file in os.listdir(subdir):
        if file.endswith(".h5ad"):
            h5ad_cf_old = os.path.join(subdir, file)
            adatas.append(sc.read(h5ad_cf_old))
adata_pred = ad.concat(adatas)

adata = sc.read_h5ad(GT_ADATA)

ap_conc_dict = {
    "XAV": [0.2, 1, 5],
    "CHIR": [0.2, 0.7, 1.5, 3.0],
    "RA": [1, 10, 100, 1000],
    "FGF8": [1, 10, 50, 100],
}
dv_conc_dict = {
    "BMP4": [5, 15, 50],
    "SHH": [1, 15, 50, 150],
}


######################
## Get interactions ##
######################


interact_df = []
for ap_mol, ap_concs in ap_conc_dict.items():
    for dv_mol, dv_concs in dv_conc_dict.items():
        for ap_conc, dv_conc in itertools.product(ap_concs, dv_concs):
            # parse perturbations
            ap_conc_idx = ap_conc_dict[ap_mol].index(ap_conc) + 1
            dv_conc_idx = dv_conc_dict[dv_mol].index(dv_conc) + 1
            ap_cond = f"{ap_mol}_{ap_conc_idx}"
            dv_cond = f"{dv_mol}_{dv_conc_idx}"
            ap_cond_pred = f"{ap_mol}_{ap_conc_idx}+ctrl"
            dv_cond_pred = f"ctrl+{dv_mol}_{dv_conc_idx}"
            comb_cond = f"{ap_mol}_{ap_conc_idx}+{dv_mol}_{dv_conc_idx}"

            # get positions in latent space
            ap_latent = adata[adata.obs.condition == ap_cond].obsm["X_pca"]
            dv_latent = adata[adata.obs.condition == dv_cond].obsm["X_pca"]
            comb_latent = adata[adata.obs.condition == comb_cond].obsm["X_pca"]
            pred_latent = adata_pred[adata_pred.obs.condition == comb_cond].obsm[
                "X_pca_reproj"
            ]

            # compute MMDs
            mmd_pred = compute_scalar_mmd(pred_latent, comb_latent)
            mmd_ap = compute_scalar_mmd(ap_latent, comb_latent)
            mmd_dv = compute_scalar_mmd(dv_latent, comb_latent)
            mmd_both = compute_scalar_mmd(
                np.concatenate([ap_latent, dv_latent], axis=0), comb_latent
            )
            mmd_ap_pred = compute_scalar_mmd(ap_latent, pred_latent)
            mmd_dv_pred = compute_scalar_mmd(dv_latent, pred_latent)
            mmd_both_pred = compute_scalar_mmd(
                np.concatenate([ap_latent, dv_latent], axis=0), pred_latent
            )

            interact_df.append(
                {
                    "ap_cond": f"{ap_mol}_{ap_conc_idx}",
                    "dv_cond": f"{dv_mol}_{dv_conc_idx}",
                    "mmd_pred_gt": mmd_pred,
                    "mmd_ap_pred": mmd_ap_pred,
                    "mmd_dv_pred": mmd_dv_pred,
                    "mmd_both_pred": mmd_both_pred,
                    "mmd_ap": mmd_ap,
                    "mmd_dv": mmd_dv,
                    "mmd_both": mmd_both,
                }
            )

            if ap_mol in ["RA", "FGF8"]:
                ap_cond = f"{ap_mol}_{ap_conc_idx}+CHIR_4"
                ap_cond_pred = f"{ap_mol}_{ap_conc_idx}+CHIR_4+ctrl"
                comb_cond = f"{ap_mol}_{ap_conc_idx}+CHIR_4+{dv_mol}_{dv_conc_idx}"
                comb_cond_pred = f"{ap_mol}_{ap_conc_idx}+CHIR_4+{dv_mol}_{dv_conc_idx}"

                ap_latent = adata[adata.obs.condition == ap_cond].obsm["X_pca"]
                comb_latent = adata[adata.obs.condition == comb_cond].obsm["X_pca"]
                pred_latent = adata_pred[
                    adata_pred.obs.condition == comb_cond_pred
                ].obsm["X_pca_reproj"]

                if TRIPLE_FROM_THREE_SINGLE:
                    ap_cond_1 = f"{ap_mol}_{ap_conc_idx}"
                    ap_cond_pred_1 = f"{ap_mol}_{ap_conc_idx}+ctrl"
                    ap_cond_2 = f"CHIR_4"
                    ap_cond_pred_2 = f"CHIR_4+ctrl"
                    ap_1_latent = adata[adata.obs.condition == ap_cond_1].obsm["X_pca"]
                    ap_2_latent = adata[adata.obs.condition == ap_cond_2].obsm["X_pca"]

                mmd_pred_gt = compute_scalar_mmd(pred_latent, comb_latent)
                mmd_ap = compute_scalar_mmd(ap_latent, comb_latent)
                mmd_dv = compute_scalar_mmd(dv_latent, comb_latent)
                mmd_ap_pred = compute_scalar_mmd(ap_latent, pred_latent)
                mmd_dv_pred = compute_scalar_mmd(dv_latent, pred_latent)

                if TRIPLE_FROM_THREE_SINGLE:
                    mmd_both = compute_scalar_mmd(
                        np.concatenate([ap_1_latent, ap_2_latent, dv_latent], axis=0),
                        comb_latent,
                    )
                    mmd_both_pred = compute_scalar_mmd(
                        np.concatenate([ap_1_latent, ap_2_latent, dv_latent], axis=0),
                        pred_latent,
                    )
                else:
                    mmd_both = compute_scalar_mmd(
                        np.concatenate([ap_latent, dv_latent], axis=0), comb_latent
                    )
                    mmd_both_pred = compute_scalar_mmd(
                        np.concatenate([ap_latent, dv_latent], axis=0), pred_latent
                    )

                interact_df.append(
                    {
                        "ap_cond": f"{ap_mol}_{ap_conc_idx}+CHIR_4",
                        "dv_cond": f"{dv_mol}_{dv_conc_idx}",
                        "mmd_pred_gt": mmd_pred_gt,
                        "mmd_ap_pred": mmd_ap_pred,
                        "mmd_dv_pred": mmd_dv_pred,
                        "mmd_both_pred": mmd_both_pred,
                        "mmd_ap": mmd_ap,
                        "mmd_dv": mmd_dv,
                        "mmd_both": mmd_both,
                    }
                )

interact_df = pd.DataFrame(interact_df)
interact_df.to_csv(os.path.join(OUTPUT_DIR, "interactions.csv"), index=False)


##########
## Plot ##
##########

# change from _CHIR at the end of ap_cond to CHIR_ at the beginning for plotting
interact_df["ap_cond"] = interact_df["ap_cond"].str.replace(
    "^(.*)\+(CHIR_4)$", r"\2_\1", regex=True
)

# deviation from both AP and DV
fig, axs = plt.subplots(1, 2, figsize=(6.5, 6.5))

ax = axs[0]
inter_gt_pivot = interact_df.pivot(
    index="ap_cond", columns="dv_cond", values="mmd_both"
)
inter_gt_pivot = inter_gt_pivot.reindex(index=inter_gt_pivot.index[::-1])
inter_pred_pivot = interact_df.pivot(
    index="ap_cond", columns="dv_cond", values="mmd_both_pred"
)
inter_pred_pivot = inter_pred_pivot.reindex(index=inter_pred_pivot.index[::-1])

sns.heatmap(
    inter_gt_pivot,
    ax=ax,
    cmap="GnBu",
    annot=False,
    fmt=".2f",
    vmin=np.nanmin(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).min(),
    vmax=np.nanmax(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).max(),
    cbar=False,
    square=True,
)

ax.set_ylabel("AP condition", fontsize=10, labelpad=10)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Ground truth")
cbar = fig.colorbar(
    ax.collections[0],
    ax=ax,
    orientation="vertical",
    ticks=[0.01, 0.03, 0.05],
    cax=fig.add_axes([1.01, 0.35, 0.03, 0.4]),
)
cbar = ax.collections[0].colorbar
ax = cbar.ax
ax.text(0.75, 0.063, "MMD", ha="center", va="center")

ax = axs[1]
sns.heatmap(
    inter_pred_pivot,
    ax=ax,
    cmap="GnBu",
    annot=False,
    fmt=".2f",
    vmin=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).min().min(),
    vmax=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).max().max(),
    cbar=False,
    square=True,
)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.set_ylabel("", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Predicted")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "interactions_both.pdf"), bbox_inches="tight", dpi=300
)
plt.close()


# deviation from AP
fig, axs = plt.subplots(1, 2, figsize=(6.5, 6.5))

ax = axs[0]
inter_gt_pivot = interact_df.pivot(index="ap_cond", columns="dv_cond", values="mmd_ap")
inter_gt_pivot = inter_gt_pivot.reindex(index=inter_gt_pivot.index[::-1])
inter_pred_pivot = interact_df.pivot(
    index="ap_cond", columns="dv_cond", values="mmd_ap_pred"
)
inter_pred_pivot = inter_pred_pivot.reindex(index=inter_pred_pivot.index[::-1])

sns.heatmap(
    inter_gt_pivot,
    ax=ax,
    cmap="YlGn",
    annot=False,
    fmt=".2f",
    vmin=np.nanmin(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).min(),
    vmax=np.nanmax(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).max(),
    cbar=False,
    square=True,
)

ax.set_ylabel("AP condition", fontsize=10, labelpad=10)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Ground truth")
cbar = fig.colorbar(
    ax.collections[0],
    ax=ax,
    orientation="vertical",
    ticks=[0.01, 0.03, 0.05, 0.07],
    cax=fig.add_axes([1.01, 0.35, 0.03, 0.4]),
)
cbar = ax.collections[0].colorbar
ax = cbar.ax
ax.text(0.75, 0.085, "MMD", ha="center", va="center")

ax = axs[1]
sns.heatmap(
    inter_pred_pivot,
    ax=ax,
    cmap="YlGn",
    annot=False,
    fmt=".2f",
    vmin=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).min().min(),
    vmax=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).max().max(),
    cbar=False,
    square=True,
)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.set_ylabel("", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Predicted")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "interactions_ap.pdf"), bbox_inches="tight", dpi=300
)
plt.close()


# deviation from DV
fig, axs = plt.subplots(1, 2, figsize=(6.5, 6.5))

ax = axs[0]
inter_gt_pivot = interact_df.pivot(index="ap_cond", columns="dv_cond", values="mmd_dv")
inter_gt_pivot = inter_gt_pivot.reindex(index=inter_gt_pivot.index[::-1])
inter_pred_pivot = interact_df.pivot(
    index="ap_cond", columns="dv_cond", values="mmd_dv_pred"
)
inter_pred_pivot = inter_pred_pivot.reindex(index=inter_pred_pivot.index[::-1])

sns.heatmap(
    inter_gt_pivot,
    ax=ax,
    cmap="OrRd",
    annot=False,
    fmt=".2f",
    vmin=np.nanmin(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).min(),
    vmax=np.nanmax(
        np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values])
    ).max(),
    cbar=False,
    square=True,
)

ax.set_ylabel("AP condition", fontsize=10, labelpad=10)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Ground truth")
cbar = fig.colorbar(
    ax.collections[0],
    ax=ax,
    orientation="vertical",
    ticks=[0.01, 0.03, 0.05, 0.07, 0.09],
    cax=fig.add_axes([1.01, 0.35, 0.03, 0.4]),
)
cbar = ax.collections[0].colorbar
ax = cbar.ax
ax.text(0.75, 0.11, "MMD", ha="center", va="center")

ax = axs[1]
sns.heatmap(
    inter_pred_pivot,
    ax=ax,
    cmap="OrRd",
    annot=False,
    fmt=".2f",
    vmin=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).min().min(),
    vmax=np.concatenate([inter_pred_pivot.values, inter_gt_pivot.values]).max().max(),
    cbar=False,
    square=True,
)
ax.set_xlabel("DV condition", fontsize=10, labelpad=10)
ax.set_ylabel("", fontsize=10, labelpad=10)
ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Predicted")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "interactions_dv.pdf"), bbox_inches="tight", dpi=300
)
plt.close()
