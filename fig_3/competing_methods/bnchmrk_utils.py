import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch


def bool2idx(x):
    """
    Returns the indices of the True-valued entries in a boolean array `x`
    """
    return np.where(x)[0]


def repeat_n(x, n):
    """
    Returns an n-times repeated version of the Tensor x,
    repetition dimension is axis 0
    """
    # copy tensor to device BEFORE replicating it n times
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def calculate_log_dose(row, molecules):
    doses = []
    for molecule in molecules:
        if molecule in row["condition"]:
            doses.append(str(row[f"{molecule}_conc"]))
    return "+".join(doses)


def get_combs_idx(adata: ad.AnnData, combs: list[str], conds: list[str]) -> np.ndarray:
    combs_idx = np.zeros(adata.n_obs, dtype=bool)
    for comb in combs:
        mols = comb.split("+") if comb != "ctrl" else [comb]
        combs_idx = combs_idx | adata.obs["comb"].apply(
            lambda x: all([mol in x for mol in mols])
            and all([mol in mols for mol in x.split("+")])
        )
    if len(conds) != 0:
        combs_idx = combs_idx | adata.obs["condition"].isin(conds)
    return combs_idx


def split_data(
    adata: ad.AnnData,
    test_dataset: str,
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, list[str]]:
    # train adata
    if exclude_combs:
        train_combs = [
            comb for comb in adata.obs["comb"].unique() if comb not in exclude_combs
        ]
    train_filt = get_combs_idx(adata, train_combs, train_conds)
    train_filt = train_filt | (adata.obs["dataset"] != test_dataset)
    adata_train = adata[train_filt]

    # eval adata
    eval_filt = get_combs_idx(adata, eval_combs, eval_conds)
    eval_filt = eval_filt & (adata.obs["dataset"] == test_dataset)
    adata_eval = adata[eval_filt]

    # sweep conditions
    sweep_filt = get_combs_idx(adata, sweep_combs, sweep_conds)
    sweep_filt = sweep_filt & (adata.obs["dataset"] == test_dataset)
    if not np.all(~sweep_filt | eval_filt):
        raise ValueError("Sweep conditions must be a subset of eval conditions.")
    sweep_conds = adata[sweep_filt].obs["condition"].unique()

    return adata_train, adata_eval, sweep_conds
