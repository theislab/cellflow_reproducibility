import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple
import cfp
import scanpy as sc
import numpy as np
import functools
from ott.solvers import utils as solver_utils
import optax
from omegaconf import OmegaConf
from typing import NamedTuple, Any
import hydra
import wandb
import anndata as ad
import pandas as pd
import os
from cfp.training import ComputationCallback
from cfp.preprocessing import transfer_labels, compute_wknn
from cfp.training import ComputationCallback
from numpy.typing import ArrayLike
from cfp.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
from cfp.metrics import compute_r_squared, compute_e_distance, compute_metrics_fast
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca


@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)

    
    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    adata = adata[adata.obs["tissue"] == "Central Nervous System"]
    ood_conds = ["cdx4_cdx1a_24"]
    control_key = config["dataset"]["control_key"]

    adata_train = adata[~adata.obs["condition"].isin(ood_conds)]
    adata_ood_true = adata[adata.obs["condition"].isin(ood_conds)]
    adata_ctrl = adata[adata.obs[control_key].to_numpy()]
    
    adata_train.uns = adata.uns.copy()
    adata_ctrl.uns = adata.uns.copy()
    adata_ood_true.uns = adata.uns.copy()

    cf = cfp.model.CellFlow(adata_train, solver="otfm")

    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    split_covariates = config_dict["dataset"]["split_covariates"]
    split_covariates = split_covariates if len(split_covariates) else None

    cf.prepare_data(
        sample_rep=config["dataset"]["sample_rep"],
        control_key=control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),
        sample_covariates=("logtime",),
        split_covariates=split_covariates,
    )

    match_fn = functools.partial(
        solver_utils.match_linear,
        epsilon=config_dict["model"]["epsilon"],
        scale_cost="mean",
        tau_a=config_dict["model"]["tau_a"],
        tau_b=config_dict["model"]["tau_b"]
    )
    optimizer = optax.MultiSteps(optax.adam(config_dict["model"]["learning_rate"]), config_dict["model"]["multi_steps"])
    flow = {config_dict["model"]["flow_type"]: config_dict["model"]["flow_noise"]}

    layers_before_pool = config_dict["model"]["layers_before_pool"]
    layers_after_pool = config_dict["model"]["layers_after_pool"]


    # Prepare the model
    cf.prepare_model(
        encode_conditions=True,
        condition_embedding_dim=config_dict["model"]["condition_embedding_dim"],
        pooling=config_dict["model"]["pooling"],
        time_encoder_dims=config_dict["model"]["time_encoder_dims"],
        time_encoder_dropout=config_dict["model"]["time_encoder_dropout"],
        hidden_dims=config_dict["model"]["hidden_dims"],
        hidden_dropout=config_dict["model"]["hidden_dropout"],
        decoder_dims=config_dict["model"]["decoder_dims"],
        decoder_dropout=config_dict["model"]["decoder_dropout"],
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        cond_output_dropout=config_dict["model"]["cond_output_dropout"],
        time_freqs=config_dict["model"]["time_freqs"],
        match_fn=match_fn,
        optimizer=optimizer,
        flow=flow,
        layer_norm_before_concatenation=config_dict["model"]["layer_norm_before_concatenation"],
        linear_projection_before_concatenation=config_dict["model"]["linear_projection_before_concatenation"],
    )

    
    wandb_callback = cfp.training.WandbLogger(project="cfp_otfm_zebrafish_first_t_cns", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    
    callbacks = [wandb_callback]
    cf.train(
        num_iterations=100000,
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=100000000,
    )

    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/first_t_cns"

    cf.save(out_dir, file_prefix=wandb.run.name)
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
