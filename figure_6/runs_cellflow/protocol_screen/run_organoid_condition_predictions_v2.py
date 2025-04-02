import warnings

warnings.filterwarnings("ignore")

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import jax.tree as jt
import jax.numpy as jnp
import cloudpickle
import itertools
import numpy as np
import scanpy as sc
from plotnine import *
import tqdm

import cfp


PLOT_DIR = "/home/fleckj/projects/cellflow/plots/organoid_cond_search/predictions/v2/"
RESULTS_DIR = (
    "/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/v2/"
)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/"
FULL_DATA_PATH = f"{DATA_DIR}/organoids_combined_full.h5ad"
adata = sc.read_h5ad(FULL_DATA_PATH)

all_datasets = adata.obs["dataset"].unique().astype(str)

cond_meta = adata.obs.iloc[:, adata.obs.columns.str.contains("_conc")].drop_duplicates()
concs_use = cond_meta.columns[(cond_meta != "0.0").sum(axis=0) > 5]
mol_concs = cond_meta[concs_use]

mol_concs[mol_concs == "100.0_250.0"] = "250.0"
mol2concs_use = {
    mol: [mol_concs[mol].astype(str).astype(float).max()] for mol in mol_concs.columns
}
mol2concs_use = {mol.replace("_conc", ""): conc for mol, conc in mol2concs_use.items()}
mol2concs = {
    mol: np.log10(np.array(concs) * 1000 + 1) for mol, concs in mol2concs_use.items()
}

#### Load cellflow model ####
# TRIAL_NAME = sys.argv[1]
TRIAL_NAME = "cellflow_0a37dcb9"
RESULTS_DIR = f"{RESULTS_DIR}/{TRIAL_NAME}/v2/predictions"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR = f"/home/fleckj/projects/cellflow/results/train_eval_organoids_common/{TRIAL_NAME}/full/model/"

cf = cfp.model.CellFlow.load(MODEL_DIR)


#### Predict for large condition space ####
from cfp.data._utils import _flatten_list

# Load encoders
mol_enc = cloudpickle.load(
    open(
        "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/mol_enc.pkl",
        "rb",
    )
)

pathway_enc = cloudpickle.load(
    open(
        "/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/pathway_enc.pkl",
        "rb",
    )
)

timings = {
    "early": [1, 8],
    "mid": [8, 15],
    "late": [15, 22],
    "early-mid": [1, 15],
    "mid-late": [8, 22],
    "early-late": [1, 22],
}

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

src_data = adata.obsm["X_latent"][np.array(adata.obs["CTRL"].values), :][:3000]

all_mols = list(mol2concs.keys())
mols2pathways = {mol: pathways[mol][0] for mol in pathways.keys()}
mols2activity = {mol: pathways[mol][1] for mol in pathways.keys()}

max_set_size = cf.data_manager.max_combination_length

all_comb_mols = list(set(mol2concs.keys()).difference(["PM"]))
combs1 = list(itertools.combinations(all_comb_mols, 1))
combs2 = list(itertools.combinations(all_comb_mols, 2))
combs3 = list(itertools.combinations(all_comb_mols, 3))
all_trial_combs = combs1 + combs2 + combs3


all_cond_reps = {}
for mol_comb in all_trial_combs:
    mol_comb = list(mol_comb)

    comb_concs = {mol: mol2concs[mol].tolist() for mol in mol_comb}
    conc_combs = list(itertools.product(*comb_concs.values()))

    for concs in conc_combs:
        for mol_timing in list(timings.keys()):

            cond_name = []
            mol_reps = {}
            for i, mol in enumerate(mol_comb):

                mol_onehot = (
                    mol_enc.transform(np.array([mol]).reshape(-1, 1))
                    .toarray()
                    .flatten()
                )

                pathway_onehot = (
                    pathway_enc.transform(np.array([mols2pathways[mol]]).reshape(-1, 1))
                    .toarray()
                    .flatten()
                )

                mol_logconc = round(concs[i], 2)
                mol_start = timings[mol_timing][0]
                mol_end = timings[mol_timing][1]

                mol_repr = mol_repr = np.concatenate(
                    (
                        mol_onehot * mol_logconc,
                        pathway_onehot,
                        [mols2activity[mol]],
                    )
                )

                mol_emb = np.concatenate(
                    (mol_repr, [float(mol_start)], [float(mol_end)])
                )
                mol_reps[mol] = mol_emb

                cond_name.append(f"{mol}_{mol_logconc}_{mol_timing}")

            # If SHH is in the combination, add PM with the same timing and conc 300
            if "SHH" in mol_comb:
                mol = "PM"
                mol_onehot = (
                    mol_enc.transform(np.array([mol]).reshape(-1, 1))
                    .toarray()
                    .flatten()
                )

                pathway_onehot = (
                    pathway_enc.transform(np.array([mols2pathways[mol]]).reshape(-1, 1))
                    .toarray()
                    .flatten()
                )

                mol_conc = 300
                mol_logconc = np.log10(mol_conc * 1000 + 1)
                mol_timing = mol_timing
                mol_start = timings[mol_timing][0]
                mol_end = timings[mol_timing][1]

                mol_repr = mol_repr = np.concatenate(
                    (
                        mol_onehot * mol_logconc,
                        pathway_onehot,
                        [mols2activity[mol]],
                    )
                )

                mol_emb = np.concatenate(
                    (mol_repr, [float(mol_start)], [float(mol_end)])
                )
                mol_reps["PM"] = mol_emb
                cond_name.append(f"PM_{mol_logconc}_{mol_timing}")

            condition_rep = jnp.expand_dims(jnp.stack(list(mol_reps.values())), 0)

            # Pad to max_set_size with zeros
            condition_rep = jnp.pad(
                condition_rep,
                ((0, 0), (0, max_set_size - condition_rep.shape[1]), (0, 0)),
                constant_values=0,
            )

            cond_name = "_".join(cond_name)
            all_cond_reps[cond_name] = condition_rep

    print(len(all_cond_reps))


condition_chunks = np.array_split(list(all_cond_reps.keys()), 300)
np.max([len(chunk) for chunk in condition_chunks])


all_preds = {}
for dataset in ["neal", "fatima", "nadya"]:
    all_preds[dataset] = {}
    dataset_rep = adata.uns["dataset_onehot"][dataset]
    # Shape into 1, comb_size, onehot_size
    dataset_rep = jnp.expand_dims(
        jnp.tile(jnp.expand_dims(dataset_rep, 0), (condition_rep.shape[1], 1)), 0
    )
    condition_data = {
        cond: {"conditions": condition_rep, "dataset": dataset_rep}
        for cond, condition_rep in all_cond_reps.items()
    }

    for chunk in tqdm.tqdm(condition_chunks):
        src_data_use = {cond: src_data for cond in chunk}
        cond_data_use = {cond: condition_data[cond] for cond in chunk}
        cond_preds = jt.map(cf.solver.predict, src_data_use, cond_data_use)
        all_preds[dataset].update(cond_preds)

    cloudpickle.dump(
        all_preds[dataset],
        open(
            f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_{dataset}.pkl",
            "wb",
        ),
    )

all_preds = {}
for dataset in all_datasets:
    cond_preds = cloudpickle.load(
        open(
            f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}_{dataset}.pkl",
            "rb",
        ),
    )
    all_preds[dataset] = cond_preds

cloudpickle.dump(
    all_preds,
    open(
        f"{RESULTS_DIR}/organoid_cond_preds_{TRIAL_NAME}.pkl",
        "wb",
    ),
)
