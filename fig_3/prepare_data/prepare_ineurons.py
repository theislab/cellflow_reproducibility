import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
import umap
import cfp


def umap_fit_transform(
    adata: ad.AnnData,
    rep: str = "X_pca",
    key_added: str = "X_umap",
) -> None:
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=500,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.4,
        spread=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        random_state=212,
        n_jobs=-1,
    )
    adata.obsm[key_added] = umap_model.fit_transform(adata.obsm[rep])


def prep_conditions(adata, ap_start, ap_end, dv_start, dv_end):
    mol_concs = {morph: [] for morph in molecules}
    mol_start_times = {morph: [] for morph in molecules}
    mol_end_times = {morph: [] for morph in molecules}
    ap_morphs = ["FGF8", "XAV", "RA", "CHIR"]
    for i, row in adata.obs.iterrows():
        mols = row[row.index.str.startswith("M_")]
        mols = mols[mols != 0]
        for mol in molecules:
            if mol == "PM":
                if "M_SHH" in mols:
                    mol_concs[mol].append(mols["M_SHH"] * 4)
                    mol_start_times[mol].append(ap_start)
                    mol_end_times[mol].append(ap_end)
                else:
                    mol_concs[mol].append(0)
                    mol_start_times[mol].append(0)
                    mol_end_times[mol].append(0)
            elif mol in mols.index.str.replace("M_", ""):
                mol_concs[mol].append(mols["M_" + mol])
                if mol in ap_morphs:
                    mol_start_times[mol].append(ap_start)
                    mol_end_times[mol].append(ap_end)
                else:
                    mol_start_times[mol].append(dv_start)
                    mol_end_times[mol].append(dv_end)
            else:
                mol_concs[mol].append(0)
                mol_start_times[mol].append(0)
                mol_end_times[mol].append(0)
    for mol in molecules:
        adata.obs[mol + "_conc"] = mol_concs[mol]
        adata.obs[mol + "_start_time"] = mol_start_times[mol]
        adata.obs[mol + "_end_time"] = mol_end_times[mol]
    return adata


def prep_cond_names(adata):
    conditions = []
    combs = []
    ap_mols = ["FGF8", "XAV", "RA", "CHIR"]
    dv_mols = ["SHH", "BMP4"]
    conc_dict = {
        mol: sorted(adata.obs[mol + "_conc"].unique().tolist()) for mol in molecules
    }
    idx = {name: i for i, name in enumerate(list(adata.obs), start=1)}
    for row in adata.obs.itertuples(name=None):
        condition = ""
        comb = ""
        for mol in ap_mols:
            if row[idx[mol + "_conc"]] > 0:
                if condition:
                    condition += "+"
                    comb += "+"
                condition += mol + "_"
                comb += mol
                condition += str(conc_dict[mol].index(row[idx[mol + "_conc"]]))
        for mol in dv_mols:
            if row[idx[mol + "_conc"]] > 0:
                if condition:
                    condition += "+"
                    comb += "+"
                condition += mol + "_"
                comb += mol
                condition += str(conc_dict[mol].index(row[idx[mol + "_conc"]]))
        condition = condition if condition else "ctrl"
        comb = comb if comb else "ctrl"
        conditions.append(condition)
        combs.append(comb)
    adata.obs["condition"] = conditions
    adata.obs["comb"] = combs
    return adata


molecules = np.array(
    [
        "FGF8",
        "XAV",
        "RA",
        "CHIR",
        "SHH",
        "BMP4",
        "SAG",
        "PM",
        "CycA",
        "FGF2",
        "FGF4",
        "FGF17",
        "FGF19",
        "BMP7",
        "LDN",
        "Activin",
        "IWP2",
        "Rspondin2",
        "Rspondin3",
        "SR11237",
        "DAPT",
        "EGF",
        "Insulin",
    ]
)
n_dims_eval = 20


##########################
### Prepare conditions ###
##########################

annot = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons/iNeurons_dr_clustered_raw_merged_meta.tsv",
    sep="\t",
)

iglut_pre = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGlut_pre_dr_clustered_raw_merged.h5ad"
)
iglut_pre_annot = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGlut_pre_dr_clustered_raw_merged_meta.tsv",
    sep="\t",
    index_col=0,
)
iglut_pre_annot.index = iglut_pre.obs.index
iglut_pre.obs = pd.concat(
    [
        iglut_pre.obs,
        iglut_pre_annot.loc[:, iglut_pre_annot.columns.str.startswith("M_")],
    ],
    axis=1,
)
iglut_pre.obs["media"] = iglut_pre_annot["Basal_media"]
annot_iglut_pre = annot[annot["sample"] == "iGlut_pre"]
iglut_pre = iglut_pre[iglut_pre.obs.index.isin(annot_iglut_pre["CellID"])]
annot_iglut_pre = annot_iglut_pre.set_index("CellID")
iglut_pre.obs["Neuron_type"] = (
    annot_iglut_pre["Neuron_type"].loc[iglut_pre.obs.index].values
)
iglut_pre.obs["Division"] = annot_iglut_pre["Division"].loc[iglut_pre.obs.index].values
iglut_pre.obs["Region"] = annot_iglut_pre["Region"].loc[iglut_pre.obs.index].values

iglut_post = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGlut_post_dr_clustered_raw_merged.h5ad"
)
iglut_post_annot = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGlut_post_dr_clustered_raw_merged_meta.tsv",
    sep="\t",
    index_col=0,
)
iglut_post_annot.index = iglut_post.obs.index
iglut_post.obs = pd.concat(
    [
        iglut_post.obs,
        iglut_post_annot.loc[:, iglut_post_annot.columns.str.startswith("M_")],
    ],
    axis=1,
)
iglut_post.obs["media"] = iglut_post_annot["Basal_media"]

annot_iglut_post = annot[np.isin(annot["sample"], ["iGlut_post_p1", "iGlut_post_p2"])]
iglut_post.obs["sample-CellID"] = (
    np.array(iglut_post.obs["sample"]) + "_" + iglut_post.obs.index
)
iglut_post = iglut_post[
    iglut_post.obs["sample-CellID"].isin(annot_iglut_post["sample-CellID"])
]
annot_iglut_post = annot_iglut_post[
    annot_iglut_post["sample-CellID"].isin(iglut_post.obs["sample-CellID"])
]
annot_iglut_post = annot_iglut_post.set_index("CellID")
iglut_post.obs["Neuron_type"] = (
    annot_iglut_post["Neuron_type"].loc[iglut_post.obs.index].values
)
iglut_post.obs["Division"] = (
    annot_iglut_post["Division"].loc[iglut_post.obs.index].values
)
iglut_post.obs["Region"] = annot_iglut_post["Region"].loc[iglut_post.obs.index].values

igaba_pre = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGABA_pre_dr_clustered_raw_merged.h5ad"
)
igaba_pre_annot = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGABA_pre_dr_clustered_raw_merged_meta.tsv",
    sep="\t",
    index_col=0,
)
igaba_pre_annot.index = igaba_pre.obs.index
igaba_pre.obs = pd.concat(
    [
        igaba_pre.obs,
        igaba_pre_annot.loc[:, igaba_pre_annot.columns.str.startswith("M_")],
    ],
    axis=1,
)
igaba_pre.obs["media"] = igaba_pre_annot["Basal_media"]
annot_igaba_pre = annot[annot["sample"] == "iGABA_pre"]
igaba_pre = igaba_pre[igaba_pre.obs.index.isin(annot_igaba_pre["CellID"])]
annot_igaba_pre = annot_igaba_pre.set_index("CellID")
igaba_pre.obs["Neuron_type"] = (
    annot_igaba_pre["Neuron_type"].loc[igaba_pre.obs.index].values
)
igaba_pre.obs["Division"] = annot_igaba_pre["Division"].loc[igaba_pre.obs.index].values
igaba_pre.obs["Region"] = annot_igaba_pre["Region"].loc[igaba_pre.obs.index].values

igaba_post = sc.read_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGABA_post_dr_clustered_raw_merged.h5ad"
)
igaba_post_annot = pd.read_csv(
    "/pmount/projects/site/pred/organoid-atlas/data/local/perturbation/MoleculeScreenHsiuChuan/iGABA_post_dr_clustered_raw_merged_meta.tsv",
    sep="\t",
    index_col=0,
)
igaba_post_annot.index = igaba_post.obs.index
igaba_post.obs = pd.concat(
    [
        igaba_post.obs,
        igaba_post_annot.loc[:, igaba_post_annot.columns.str.startswith("M_")],
    ],
    axis=1,
)
igaba_post.obs["media"] = igaba_post_annot["Basal_media"]
annot_igaba_post = annot[annot["sample"] == "iGABA_post"]
igaba_post = igaba_post[igaba_post.obs.index.isin(annot_igaba_post["CellID"])]
annot_igaba_post = annot_igaba_post.set_index("CellID")
igaba_post.obs["Neuron_type"] = (
    annot_igaba_post["Neuron_type"].loc[igaba_post.obs.index].values
)
igaba_post.obs["Division"] = (
    annot_igaba_post["Division"].loc[igaba_post.obs.index].values
)
igaba_post.obs["Region"] = annot_igaba_post["Region"].loc[igaba_post.obs.index].values

iglut_pre = prep_conditions(iglut_pre, 1, 5, 2, 5)
iglut_post = prep_conditions(iglut_post, 1, 10, 2, 10)
igaba_pre = prep_conditions(igaba_pre, 1, 5, 2, 5)
igaba_post = prep_conditions(igaba_post, 4, 15, 5, 15)


##########################
### Glut post dataset ####
##########################


def prep_adata(adata):
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=4000, subset=True
    )
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cfp.pp.centered_pca(adata, n_comps=n_dims_eval, method="rapids")
    umap_fit_transform(adata, rep="X_pca", key_added="X_umap")
    return adata


iglut_post_only = prep_adata(iglut_post)
iglut_post_only = prep_cond_names(iglut_post_only)
iglut_post_only.obs["dataset"] = "glut_post"
iglut_post_only.write_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons/ineurons_glutpost.h5ad"
)


################################
### All iNeurons, joint HVGs ###
################################

ineur_all = ad.concat([iglut_pre, iglut_post, igaba_pre, igaba_post], join="outer")
ineur_all.obs_names_make_unique()
sc.pp.highly_variable_genes(
    ineur_all, flavor="seurat_v3", n_top_genes=4000, subset=True
)
ineur_all.layers["counts"] = ineur_all.X.copy()
sc.pp.normalize_total(ineur_all, target_sum=1e4)
sc.pp.log1p(ineur_all)
# ineur_all.obs = ineur_all.obs.drop(columns=[col for col in ineur_all.obs.columns if not np.any(["_start_time" in col, "_end_time" in col, "_conc" in col, "media" in col])])
ineur_all = prep_cond_names(ineur_all)
ineur_all.obs["dataset"] = (
    ["glut_pre"] * iglut_pre.n_obs
    + ["glut_post"] * iglut_post.n_obs
    + ["gaba_pre"] * igaba_pre.n_obs
    + ["gaba_post"] * igaba_post.n_obs
)
cfp.pp.centered_pca(ineur_all, n_comps=n_dims_eval, method="rapids")
umap_fit_transform(ineur_all, rep="X_pca", key_added="X_umap")

ineur_all.write_h5ad(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/data/ineurons/ineurons_jointhvg.h5ad"
)
