import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple

import cellflow
import scanpy as sc
import numpy as np
import functools
from ott.solvers import utils as solver_utils
import optax
from omegaconf import OmegaConf
from typing import NamedTuple, Any
import hydra
import wandb




@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    split = config_dict["dataset"]["split"]
    adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
    adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
    adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
    adata_train = sc.read_h5ad(adata_train_path)
    adata_test = sc.read_h5ad(adata_test_path)
    adata_ood = sc.read_h5ad(adata_ood_path)
    
    adata_ood_1 = adata_ood[adata_ood.obs["dose"].isin((0.0, 10.0))]
    adata_ood_2 = adata_ood[adata_ood.obs["dose"].isin((0.0, 100.0))]
    adata_ood_3 = adata_ood[adata_ood.obs["dose"].isin((0.0, 1000.0))]
    adata_ood_4 = adata_ood[adata_ood.obs["dose"].isin((0.0, 10000.0))]
    

    cf = cellflow.model.CellFlow(adata_train, solver="otfm")

    # Prepare the training data and perturbation conditions
    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    sample_covariates = list(config_dict["dataset"]["sample_covariates"]) if list(config_dict["dataset"]["sample_covariates"])[0] is not None else None
    sample_covariate_reps = dict(config_dict["dataset"]["sample_covariate_reps"]) if sample_covariates is not None else None
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),
        sample_covariates=sample_covariates,
        sample_covariate_reps=sample_covariate_reps,
        split_covariates=list(config_dict["dataset"]["split_covariates"]),
    )

    match_fn = functools.partial(
        solver_utils.match_linear,
        epsilon=config_dict["model"]["epsilon"],
        scale_cost="mean",
        tau_a=config_dict["model"]["tau_a"],
        tau_b=config_dict["model"]["tau_b"]
    )
    optimizer = optax.MultiSteps(optax.adam(config_dict["model"]["learning_rate"]), config_dict["model"]["multi_steps"])
    probability_path = {config_dict["model"]["flow_type"]: config_dict["model"]["flow_noise"]}

    layers_before_pool = config_dict["model"]["layers_before_pool"]
    layers_after_pool = config_dict["model"]["layers_after_pool"]

    solver_kwargs = {"ema": config_dict["model"]["ema"]}
    #time_max_period = config_dict["model"]["time_max_period"]
    #time_max_period = time_max_period if time_max_period>0 else None

    # Prepare the model
    cf.prepare_model(
        condition_mode="deterministic",
        regularization=config_dict["model"]["regularization"],
        pooling=config_dict["model"]["pooling"],
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=config_dict["model"]["condition_embedding_dim"],
        cond_output_dropout=config_dict["model"]["cond_output_dropout"],
        time_freqs=config_dict["model"]["time_freqs"],
        time_max_period=None,
        time_encoder_dims=config_dict["model"]["time_encoder_dims"],
        time_encoder_dropout=config_dict["model"]["time_encoder_dropout"],
        hidden_dims=config_dict["model"]["hidden_dims"],
        hidden_dropout=config_dict["model"]["hidden_dropout"],
        conditioning=config_dict["model"]["conditioning"],
        decoder_dims=config_dict["model"]["decoder_dims"],
        decoder_dropout=config_dict["model"]["decoder_dropout"],
        probability_path=probability_path,
        solver_kwargs=solver_kwargs,
        match_fn=match_fn,
        optimizer=optimizer,
        layer_norm_before_concatenation=config_dict["model"]["layer_norm_before_concatenation"],
        linear_projection_before_concatenation=config_dict["model"]["linear_projection_before_concatenation"],
    )

    cf.prepare_validation_data(
        adata_ood_1,
        name="ood_1",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_2,
        name="ood_2",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_3,
        name="ood_3",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_4,
        name="ood_4",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )


    metrics_callback = cellflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cellflow.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
    wandb_callback = cellflow.training.WandbLogger(project="cellflow_from_before", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)

    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]

    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    #if config_dict["training"]["save_model"]:
    #    cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
