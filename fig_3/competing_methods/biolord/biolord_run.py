import sys
import os
import warnings
import yaml
from typing import Literal
from functools import reduce
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
import biolord
import inspect

sys.path.append(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/otfm/run/"
)
from bnchmrk_utils import *


warnings.filterwarnings("ignore")


def run_biolord_ineurons(
    *,
    h5ad: str,
    train_datasets: list[Literal["glut_post", "glut_pre", "gaba_post", "gaba_pre"]],
    test_dataset: Literal["glut_post", "gaba_post"],
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
    out_dir: str,
    out_prefix: str,
    save_model: bool,
    max_epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    reconstruction_penalty: float,
    unknown_attribute_penalty: float,
    unknown_attribute_noise_param: float,
    attribute_dropout_rate: float,
    use_batch_norm: bool,
    use_layer_norm: bool,
    n_latent: int,
    decoder_width: int,
    decoder_depth: int,
    attribute_nn_width: int,
    attribute_nn_depth: int,
    n_latent_attribute_ordered: int,
) -> None:

    ####################
    #### Load data #####
    ####################

    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    adata = sc.read_h5ad(h5ad)
    adata = adata[adata.obs["dataset"].isin(train_datasets)]

    for mol in molecules:
        adata.obs[f"{mol}_conc"] = np.log1p(adata.obs[f"{mol}_conc"])

    ####################
    ### Prepare data ###
    ####################

    # split data
    adata_train, adata_eval, sweep_conds = split_data(
        adata,
        test_dataset,
        train_combs,
        exclude_combs,
        eval_combs,
        sweep_combs,
        train_conds,
        eval_conds,
        sweep_conds,
    )

    ####################
    ### Prepare train ##
    ####################

    biolord_params = {
        "decoder_width": decoder_width,
        "decoder_depth": decoder_depth,
        "attribute_nn_width": attribute_nn_width,
        "attribute_nn_depth": attribute_nn_depth,
        "n_latent_attribute_ordered": n_latent_attribute_ordered,
        "gene_likelihood": "normal",
        "reconstruction_penalty": reconstruction_penalty,
        "unknown_attribute_penalty": unknown_attribute_penalty,
        "unknown_attribute_noise_param": unknown_attribute_noise_param,
        "attribute_dropout_rate": attribute_dropout_rate,
        "use_batch_norm": use_batch_norm,
        "use_layer_norm": use_layer_norm,
        "seed": 42,
    }

    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }

    ####################
    ###### Run OT ######
    ####################

    adata.obs["train_test_split"] = np.where(
        adata.obs.index.isin(adata_train.obs.index), "train", "test"
    )
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=[mol + "_conc" for mol in molecules],
        categorical_attributes_keys=None,
    )
    model = biolord.Biolord(
        adata=adata,
        n_latent=n_latent,
        model_name=f"biolord_{out_prefix}",
        module_params=biolord_params,
        train_classifiers=False,
        split_key="train_test_split",
    )
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=early_stopping_patience,
        check_val_every_n_epoch=10,
        num_workers=1,
        enable_checkpointing=False,
    )

    ####################
    #### Evaluate OT ###
    ####################

    layer = "X"
    n_obs = 977
    pert_categories_index = pd.Index(
        adata_eval.obs["condition"].values, dtype="category"
    )

    dataset_control = model.get_dataset(adata[adata.obs["comb"] == "ctrl"])
    dataset_eval = model.get_dataset(adata[adata.obs_names.isin(adata_eval.obs_names)])

    adatas_pred = []
    np.random.seed(212)
    for condition in adata_eval.obs["condition"].unique():
        bool_category = pert_categories_index.get_loc(condition)
        idx_all = bool2idx(bool_category)
        idx = idx_all[0]

        dataset_comb = {}
        dataset_comb[layer] = dataset_control[layer].to(model.device)
        dataset_comb["ind_x"] = dataset_control["ind_x"].to(model.device)
        for key in dataset_control:
            if key not in [layer, "ind_x"]:
                dataset_comb[key] = repeat_n(dataset_eval[key][idx, :], n_obs)

        pred_mean, pred_std = model.module.get_expression(dataset_comb)
        pred_mean, pred_std = (
            pred_mean.cpu().detach().numpy(),
            pred_std.cpu().detach().numpy(),
        )
        X_pred = np.random.normal(pred_mean, pred_std)

        adata_pred = ad.AnnData(X=csr_matrix(X_pred))
        adata_pred.obs["dataset"] = "glut_post"
        adata_pred.obs["comb"] = adata_eval.obs["comb"][
            adata_eval.obs["condition"] == condition
        ].values[0]
        adata_pred.obs["condition"] = condition
        adatas_pred.append(adata_pred)
    adata_pred = ad.concat(adatas_pred, join="outer")

    # save outputs
    if save_model:
        model_path = os.path.join(out_dir, f"{out_prefix}_model")
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path, overwrite=True)
    adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    fn_sig = inspect.signature(run_biolord_ineurons)
    config = {k: v for k, v in config.items() if k in fn_sig.parameters}
    run_biolord_ineurons(**config)
