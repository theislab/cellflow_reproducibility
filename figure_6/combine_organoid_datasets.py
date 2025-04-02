import cloudpickle
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd

import sklearn.preprocessing as preprocessing

import cfp
from cfp.data._utils import encode_onehot


#### Read all organoid datasets ####
adata_midbrain = sc.read_h5ad(
    "/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v2_common_hv2k_wknn.h5ad"
)
adata_amin = sc.read_h5ad(
    "/projects/site/pred/organoid-atlas/data/public_datasets/scg/brain_organoids/AminPasca2023brx/pasca_primary_screen_v1_common_hv2k_wknn.h5ad"
)
adata_fatima = sc.read_h5ad(
    "/projects/site/pred/organoid-atlas/data/local/perturbation/PatterningScreenFatima/fatima_patscreen_v1_common_hv2k_wknn.h5ad"
)

adata_midbrain.obs.index = adata_midbrain.obs["original_name"].values
adata_amin_obs_index = (
    "AminPasca_"
    + adata_amin.obs["morph_comb"].str.replace("+", "_").values
    + "_"
    + np.arange(adata_amin.shape[0]).astype(str)
)
adata_amin.obs.index = adata_amin_obs_index

adata_midbrain.write_h5ad(
    "/projects/site/pred/organoid-atlas/data/local/perturbation/MidbrainPatterningNadya/midbrain_patterning_v3index_common_hv2k_wknn.h5ad"
)
adata_amin.write_h5ad(
    "/projects/site/pred/organoid-atlas/data/public_datasets/scg/brain_organoids/AminPasca2023brx/pasca_primary_screen_v2index_common_hv2k_wknn.h5ad"
)
adata_fatima.write_h5ad(
    "/projects/site/pred/organoid-atlas/data/local/perturbation/PatterningScreenFatima/fatima_patscreen_v2index_common_hv2k_wknn.h5ad"
)


#### Get harmonized protocol conditions ####
conditions_df = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/USERS/jfleck/data/organoids_combined/organoids_conditions.tsv",
    sep="\t",
)


#### Combine all datasets ####
combined_adata = ad.concat(
    [
        adata_fatima,
        adata_amin,
        adata_midbrain,
    ],
    join="outer",
)

combined_adata.obs = conditions_df

combined_adata.obsm["X_latent"] = np.concatenate(
    [
        adata_fatima.obsm["X_scanvi_braun"],
        adata_amin.obsm["X_scanvi_braun"],
        adata_midbrain.obsm["X_scanvi_braun"],
    ]
)

combined_adata.obs["dataset"] = np.concatenate(
    [
        ["fatima"] * adata_fatima.shape[0],
        ["neal"] * adata_amin.shape[0],
        ["nadya"] * adata_midbrain.shape[0],
    ]
)

combined_adata.write_h5ad(
    "/projects/site/pred/organoid-atlas/data/organoids_combined/organoids_combined_full.h5ad"
)


#### Condition processing ####
combined_adata = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v2/organoids_combined_full.h5ad"
)


#### Annotate molecules with pathways and activities ####
molecules = (
    combined_adata.obs.columns[combined_adata.obs.columns.str.contains("_conc")]
    .str.split("_")
    .str[0]
    .unique()
    .values
)

pathways = {
    "SAG": ("SHH", 1),
    "SHH": ("SHH", 1),
    "PM": ("SHH", 1),
    "CycA": ("SHH", -1),
    "FGF2": ("FGF", 1),
    "FGF4": ("FGF", 1),
    "FGF8": ("FGF", 1),
    "FGF17": ("FGF", 1),
    "FGF19": ("FGF", 1),
    "BMP4": ("TGFb", 1),
    "BMP7": ("TGFb", 1),
    "LDN": ("TGFb", -1),
    "Activin": ("TGFb", 1),
    "CHIR": ("WNT", 1),
    "XAV": ("WNT", -1),
    "IWP2": ("WNT", -1),
    "Rspondin2": ("WNT", 1),
    "Rspondin3": ("WNT", 1),
    "RA": ("RA", 1),
    "SR11237": ("RA", 1),
    "DAPT": ("Notch", -1),
    "EGF": ("EGF", 1),
    "Insulin": ("Insulin", 1),
}

mols2pathways = {mol: pathways[mol][0] for mol in molecules}
mols2activity = {mol: pathways[mol][1] for mol in molecules}


##### Prepare condition representations with pathways ####
mol_enc = preprocessing.OneHotEncoder()
mol_enc.fit(molecules.reshape(-1, 1))

pathway_enc = preprocessing.OneHotEncoder()
pathway_enc.fit(np.array(list(set(mols2pathways.values()))).reshape(-1, 1))

# save encoders
cloudpickle.dump(
    mol_enc,
    open(
        "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/mol_enc.pkl",
        "wb",
    ),
)
cloudpickle.dump(
    pathway_enc,
    open(
        "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/pathway_enc.pkl",
        "wb",
    ),
)
cloudpickle.dump(
    mols2pathways,
    open(
        "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/mols2pathways.pkl",
        "wb",
    ),
)

combined_adata.uns["conditions"] = {}
condition_keys = []
for mol in molecules:
    conditions = combined_adata.obs[
        [mol + "_conc", mol + "_start_time", mol + "_end_time"]
    ].drop_duplicates()
    for i, cond in conditions.iterrows():
        if cond.iloc[0] == "0.0":
            continue
        cond_cells = (
            (combined_adata.obs[mol + "_conc"] == cond.iloc[0])
            & (combined_adata.obs[mol + "_start_time"] == cond.iloc[1])
            & (combined_adata.obs[mol + "_end_time"] == cond.iloc[2])
        )

        mol_onehot = (
            mol_enc.transform(np.array([mol]).reshape(-1, 1)).toarray().flatten()
        )

        pathway_onehot = (
            pathway_enc.transform(np.array([mols2pathways[mol]]).reshape(-1, 1))
            .toarray()
            .flatten()
        )

        concs = cond.iloc[0].split("_")
        start_times = cond.iloc[1].split("_")
        end_times = cond.iloc[2].split("_")
        for conc, start, end in zip(concs, start_times, end_times):
            cond_id = mol + "_" + conc + "_" + start + "_" + end
            if cond_id in condition_keys:
                continue
            condition_keys.append(cond_id)
            combined_adata.obs[cond_id] = cond_cells
            logconc = np.log10(float(conc) * 1000 + 1)

            mol_repr = mol_repr = np.concatenate(
                (mol_onehot * logconc, pathway_onehot, [mols2activity[mol]])
            )

            combined_adata.uns["conditions"][cond_id] = np.concatenate(
                (mol_repr, [float(start)], [float(end)])
            )


# add one-hot encoding of datasets / base protocols
cfp.pp.encode_onehot(
    combined_adata,
    covariate_keys=["dataset"],
    exclude_values=["CTRL"],
    uns_key_added="dataset_onehot",
)


combined_adata.write_h5ad(
    "/projects/site/pred/organoid-atlas/data/organoids_combined/organoids_combined_full.h5ad"
)


#### Fix source distributions ####
condition_uns = combined_adata.uns

combined_adata_noctrl = combined_adata[~combined_adata.obs["CTRL"]].copy()
combined_adata_ctrl = combined_adata[combined_adata.obs["CTRL"]].copy()

combined_adata_ctrl_amin = combined_adata_ctrl.copy()
combined_adata_ctrl_amin.obs["dataset"] = "neal"
combined_adata_ctrl_amin.obs_names = "amin_" + combined_adata_ctrl_amin.obs.index

combined_adata_ctrl_fatima = combined_adata_ctrl.copy()
combined_adata_ctrl_fatima.obs["dataset"] = "fatima"
combined_adata_ctrl_fatima.obs_names = "fatima_" + combined_adata_ctrl_fatima.obs.index

combined_adata_ctrl_nadya = combined_adata_ctrl.copy()
combined_adata_ctrl_nadya.obs["dataset"] = "nadya"
combined_adata_ctrl_nadya.obs_names = "nadya_" + combined_adata_ctrl_nadya.obs.index

combined_adata_ctrl = ad.concat(
    {
        "neal": combined_adata_ctrl_amin,
        "fatima": combined_adata_ctrl_fatima,
        "nadya": combined_adata_ctrl_nadya,
    }
)

combined_adata = ad.concat([combined_adata_noctrl, combined_adata_ctrl])
combined_adata.uns = condition_uns

combined_adata.write_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/organoids_combined_full.h5ad"
)
