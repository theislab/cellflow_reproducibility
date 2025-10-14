import sys
import os
import warnings
import yaml
from typing import Literal
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
import pandas as pd
import cpa
import inspect

sys.path.append(
    "/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/"
)
from bnchmrk_utils import *


warnings.filterwarnings("ignore")


def run_cpa_ineurons(
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
    use_counts: bool,
    max_epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    n_epochs_kl_warmup: int,
    n_epochs_pretrain_ae: int,
    n_epochs_adv_warmup: int,
    n_epochs_mixup_warmup: int,
    mixup_alpha: float,
    adv_steps: int,
    n_hidden_adv: int,
    n_layers_adv: int,
    use_batch_norm_adv: bool,
    use_layer_norm_adv: bool,
    dropout_rate_adv: float,
    reg_adv: float,
    pen_adv: float,
    n_latent: int,
    doser_type: Literal["logsigm", "sigm", "linear"],
    n_hidden_encoder: int,
    n_layers_encoder: int,
    n_hidden_decoder: int,
    n_layers_decoder: int,
    use_batch_norm: bool,
    use_layer_norm: bool,
    dropout_rate_encoder: float,
    dropout_rate_decoder: float,
    variational: bool,
) -> None:

    ####################
    #### Load data #####
    ####################

    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    adata = sc.read_h5ad(h5ad)
    adata = adata[adata.obs["dataset"].isin(train_datasets)]

    ####################
    ### Prepare data ###
    ####################

    # prepare conditions
    for mol in molecules:
        adata.obs[f"{mol}_conc"] = np.log1p(adata.obs[f"{mol}_conc"])
    adata.obs["condition_original"] = adata.obs["condition"].copy()
    adata.obs["condition"] = adata.obs["condition"].str.replace("_CHIR", "+CHIR")
    adata.obs["condition"] = adata.obs["condition"].str.replace("+ctrl", "")
    adata.obs["condition"] = adata.obs["condition"].str.replace("ctrl+", "")
    adata.obs["log_dose"] = adata.obs.apply(
        lambda row: calculate_log_dose(row, molecules), axis=1
    )
    adata.obs.loc[adata.obs["condition"] == "ctrl", "log_dose"] = "0"

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

    # choose counts or log1p
    if use_counts:
        adata.X = adata.layers["counts"].copy()

    ####################
    ### Prepare train ##
    ####################

    ae_hparams = {
        "n_latent": n_latent,
        "recon_loss": "nb" if use_counts else "gauss",
        "doser_type": doser_type,
        "n_hidden_encoder": n_hidden_encoder,
        "n_layers_encoder": n_layers_encoder,
        "n_hidden_decoder": n_hidden_decoder,
        "n_layers_decoder": n_layers_decoder,
        "use_batch_norm_encoder": use_batch_norm,
        "use_layer_norm_encoder": use_layer_norm,
        "use_batch_norm_decoder": use_batch_norm,
        "use_layer_norm_decoder": use_layer_norm,
        "dropout_rate_encoder": dropout_rate_encoder,
        "dropout_rate_decoder": dropout_rate_decoder,
        "variational": variational,
        "seed": 212,
    }

    trainer_params = {
        "n_epochs_kl_warmup": n_epochs_kl_warmup,
        "n_epochs_pretrain_ae": n_epochs_pretrain_ae,
        "n_epochs_adv_warmup": n_epochs_adv_warmup,
        "n_epochs_mixup_warmup": n_epochs_mixup_warmup,
        "mixup_alpha": mixup_alpha,
        "adv_steps": adv_steps,
        "n_hidden_adv": n_hidden_adv,
        "n_layers_adv": n_layers_adv,
        "use_batch_norm_adv": use_batch_norm_adv,
        "use_layer_norm_adv": use_layer_norm_adv,
        "dropout_rate_adv": dropout_rate_adv,
        "reg_adv": reg_adv,
        "pen_adv": pen_adv,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": False,
        "gradient_clip_value": 1.0,
        "step_size_lr": 45,
    }

    ####################
    ###### Run OT ######
    ####################

    adata.obs["train_test_split"] = "test"
    adata.obs.loc[adata_train.obs_names, "train_test_split"] = "train"
    adata.obs.loc[
        np.random.choice(
            adata_train.obs_names, size=adata_train.n_obs // 10, replace=False
        ),
        "train_test_split",
    ] = "valid"

    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        dosage_key="log_dose",
        control_group="ctrl",
        batch_key=None,
        max_comb_len=3,
        is_count_data=use_counts,
    )
    model = cpa.CPA(
        adata=adata,
        split_key="train_test_split",
        train_split="train",
        valid_split="valid",
        test_split="test",
        **ae_hparams,
    )

    model_path = os.path.join(out_dir, f"{out_prefix}_model")
    os.makedirs(model_path, exist_ok=True)
    model.train(
        max_epochs=max_epochs,
        use_gpu=True,
        batch_size=batch_size,
        plan_kwargs=trainer_params,
        early_stopping_patience=early_stopping_patience,
        check_val_every_n_epoch=5,
        save_path=model_path if save_model else None,
    )

    ####################
    #### Evaluate OT ###
    ####################

    # plot latent space
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    sc.settings.figdir = figs_dir
    latent_outputs = model.get_latent_representation(adata, batch_size=1024)
    latent_basal_adata = latent_outputs["latent_basal"]
    latent_adata = latent_outputs["latent_after"]

    sc.pp.neighbors(latent_basal_adata)
    sc.tl.umap(latent_basal_adata)
    sc.pl.umap(
        latent_basal_adata,
        color=["comb"],
        frameon=False,
        wspace=0.2,
        palette="tab20",
        save="_basal.pdf",
    )

    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(
        latent_adata,
        color=["comb"],
        frameon=False,
        wspace=0.2,
        palette="tab20",
        save="_after.pdf",
    )

    adatas_cpa = []
    for cond in adata_eval.obs["condition"].unique():
        adata_cpa = adata[adata.obs["comb"] == "ctrl"].copy()
        adata_cpa_obs = pd.DataFrame(
            index=adata[adata.obs["comb"] == "ctrl"].obs_names,
        )
        adata_cpa_obs["condition"] = cond
        adata_cpa_obs["condition_original"] = (
            adata[adata.obs["condition"] == cond].obs["condition_original"].values[0]
        )
        adata_cpa_obs["comb"] = (
            adata[adata.obs["condition"] == cond].obs["comb"].values[0]
        )
        adata_cpa_obs["dataset"] = (
            adata[adata.obs["condition"] == cond].obs["dataset"].values[0]
        )
        adata_cpa_obs["log_dose"] = (
            adata[adata.obs["condition"] == cond].obs["log_dose"].values[0]
        )
        adata_cpa.obs = adata_cpa_obs
        adatas_cpa.append(adata_cpa)

    adata_cpa = ad.concat(adatas_cpa)

    cpa.CPA.setup_anndata(
        adata_cpa,
        perturbation_key="condition",
        dosage_key="log_dose",
        control_group="ctrl",
        batch_key=None,
        max_comb_len=3,
        is_count_data=use_counts,
    )
    model.predict(adata_cpa, batch_size=1024)

    x_pred = adata_cpa.obsm["CPA_pred"]
    adata_pred = ad.AnnData(X=csr_matrix(x_pred))

    if use_counts:
        adata_pred.layers["counts"] = adata_pred.X.copy()
        sc.pp.normalize_total(adata_pred, target_sum=1e4)
        sc.pp.log1p(adata_pred)

    adata_pred.obs["dataset"] = "glut_post"
    adata_pred.obs["condition_cpa"] = adata_cpa.obs["condition"].values
    adata_pred.obs["condition"] = adata_cpa.obs["condition_original"].values
    adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")


if __name__ == "__main__":
    # load config and run OT-FM
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    fn_sig = inspect.signature(run_cpa_ineurons)
    config = {k: v for k, v in config.items() if k in fn_sig.parameters}
    run_cpa_ineurons(**config)
