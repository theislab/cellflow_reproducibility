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
import pickle
import pandas as pd
import anndata as ad

class PCADecoder(NamedTuple):
    pcs: Any
    means: Any




def prepare_data(adata_train, adata_test, adata_ood, path_dict):

    adata_train.obs['control'] = False
    adata_test.obs['control'] = False
    adata_ood.obs['control'] = False

    adata_train.obs['logtimepoint'] = np.log(pd.to_numeric(adata_train.obs['timepoint']))
    adata_test.obs['logtimepoint'] = np.log(pd.to_numeric(adata_test.obs['timepoint']))
    adata_ood.obs['logtimepoint'] = np.log(pd.to_numeric(adata_ood.obs['timepoint']))
    
    adata_train.obs['logtimepoint2'] = adata_train.obs['logtimepoint']
    adata_test.obs['logtimepoint2'] = adata_test.obs['logtimepoint']
    adata_ood.obs['logtimepoint2'] = adata_ood.obs['logtimepoint']

    adata_train.obs.loc[(adata_train.obs['gene1+gene2'] == 'negative+negative') & (adata_train.obs['timepoint'] == '18'), 'control'] = True 

    adata_test.obs.loc[(adata_test.obs['gene1+gene2'] == 'negative+negative') & (adata_test.obs['timepoint'] == '18'), 'control'] = True 

    adata_ood.obs.loc[(adata_ood.obs['gene1+gene2'] == 'negative+negative') & (adata_ood.obs['timepoint'] == '18'), 'control'] = True 
    
    adata_ood = ad.concat((adata_ood, adata_test[adata_test.obs.control == True])) # add controls to ood

    file_path = os.path.join(path_dict)

    with open(file_path, 'rb') as file:
        gene_dict = pickle.load(file) 

    adata_train.uns['crispr_rep'] = gene_dict
    adata_test.uns['crispr_rep'] = gene_dict
    adata_ood.uns['crispr_rep'] = gene_dict

    return adata_train, adata_test, adata_ood


@hydra.main(config_path="conf", config_name="train_zebrafish")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    adata_train = sc.read_h5ad(config_dict["dataset"]["adata_train_path"])
    adata_test = sc.read_h5ad(config_dict["dataset"]["adata_test_path"])
    adata_ood = sc.read_h5ad(config_dict["dataset"]["adata_ood_path"])
    
    del adata_train.obsm['emb_1'], adata_train.obsm['emb_2'], adata_test.obsm['emb_1'], adata_test.obsm['emb_2'], adata_ood.obsm['emb_1'], adata_ood.obsm['emb_2']
    
    adata_train.varm["X_mean"] = adata_train.varm["X_train_mean"]
    adata_test.varm["X_mean"] = adata_test.varm["X_train_mean"]
    adata_ood.varm["X_mean"] = adata_ood.varm["X_train_mean"]

    del adata_train.varm["X_train_mean"], adata_test.varm["X_train_mean"], adata_ood.varm["X_train_mean"]
        
    adata_train, adata_test, adata_ood = prepare_data(adata_train, adata_test, adata_ood, config_dict["dataset"]["embedding_dict"])
    
    adata_train.obsm["X_pca_use"] = adata_train.obsm["X_pca"][:, :config_dict["dataset"]["pca_dims"]]
    adata_test.obsm["X_pca_use"] = adata_test.obsm["X_pca"][:, :config_dict["dataset"]["pca_dims"]]
    adata_ood.obsm["X_pca_use"] = adata_ood.obsm["X_pca"][:, :config_dict["dataset"]["pca_dims"]]

    cf = cfp.model.CellFlow(adata_train, solver="otfm")

    # Prepare the training data and perturbation conditions
    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()} # gene1, gene2, logtime
    cf.prepare_data(
        sample_rep="X_pca_use",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),    )

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
        time_encoder_dims=config_dict["model"]["time_encoder_dims"],
        time_encoder_dropout=config_dict["model"]["time_encoder_dropout"],
        hidden_dims=config_dict["model"]["hidden_dims"],
        hidden_dropout=config_dict["model"]["hidden_dropout"],
        decoder_dims=config_dict["model"]["decoder_dims"],
        decoder_dropout=config_dict["model"]["decoder_dropout"],
        pooling=config_dict["model"]["pooling"],
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        cond_output_dropout=config_dict["model"]["cond_output_dropout"],
        time_freqs=config_dict["model"]["time_freqs"],
        match_fn=match_fn,
        optimizer=optimizer,
        flow=flow,
    )

    cf.prepare_validation_data(
        adata_test,
        name="test",
        n_conditions_on_log_iteration=config_dict["training"]["test_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["test_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood,
        name="ood",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    #metrics_callback = cfp.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    metrics_callback = cfp.training.Metrics(metrics=["r_squared", "mmd", "e_distance"], sample_size=5000)
    decoded_metrics_callback = cfp.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
    #sampled_metrics_callback = cfp.training.SampledMetrics(metrics=["mmd", "e_distance"], sample_size=5000)
    wandb_callback = cfp.training.WandbLogger(project="otfm-zebrafish", out_dir="/home/icb/alejandro.tejada/tmp", config=config_dict)

    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]

    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    
    # Create a distinctive filename
    model_name = f"zebrafish_model_{config_dict['training']['num_iterations']}iters_{config_dict['model']['condition_embedding_dim']}dim"
    save_path = os.path.join(config_dict["training"]["save_path"], model_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # cf.save(save_path)

    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)