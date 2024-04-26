# adapted https://github.com/nitzanlab/biolord_reproducibility/blob/main/utils/utils_perturbations_sciplex3.py

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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


def compute_prediction(
    model,
    adata,
    dataset,
    dataset_control
):
    pert_categories_index = pd.Index(adata.obs["cov_drug_dose_name"].values, dtype="category")

    cl_dict = {
        torch.Tensor([0.]): "A549",
        torch.Tensor([1.]): "K562",
        torch.Tensor([2.]): "MCF7",
    }

    cell_lines = ["A549", "K562", "MCF7"]
    layer = "X" if "X" in dataset else "layers"
    pred_output = {}
    for cell_drug_dose_comb, _ in tqdm(
        zip(*np.unique(pert_categories_index.values, return_counts=True))
    ):
        bool_category = pert_categories_index.get_loc(cell_drug_dose_comb)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]

        dataset_comb = {}

        n_obs = dataset_control[layer].size(0)
        dataset_comb[layer] = dataset_control[layer].to(model.device)
        dataset_comb["ind_x"] = dataset_control["ind_x"].to(model.device)
        for key in dataset_control:
            if key not in [layer, "ind_x"]:
                dataset_comb[key] = repeat_n(dataset[key][idx, :], n_obs)

        stop = False
        for tensor, cl in cl_dict.items():
            if (tensor == dataset["cell_type"][idx]).all():
                if cl not in cell_lines:
                    stop = True
        if stop:
            continue

        pred, _ = model.module.get_expression(dataset_comb)
        pred_output[cell_drug_dose_comb] = pred
    return pred_output
