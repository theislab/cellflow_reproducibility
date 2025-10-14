import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple
import pickle
import cellflow
from cellflow import preprocessing as cfpp
import scanpy as sc
import numpy as np
import functools
from ott.solvers import utils as solver_utils
import optax
from omegaconf import OmegaConf
from typing import NamedTuple, Any
import hydra
import wandb
import pandas as pd
import time
import anndata as ad

from numpy.typing import ArrayLike


def add_subgroup_annotations(adata_train, adata): 

    train_conditions = adata_train.obs.condition.str.replace("+ctrl", "").str.replace("ctrl+", "").unique()

    assert not adata[adata.obs.condition != "ctrl"].obs.condition.isin(train_conditions).any()

    mask_single_perturbation = adata.obs.condition.str.contains("ctrl")
    mask_double_perturbation_seen_0 = (
        ~adata.obs.condition.str.contains("ctrl") & 
        ~adata.obs.gene_1.isin(train_conditions) & 
        ~adata.obs.gene_2.isin(train_conditions)
    )
    mask_double_perturbation_seen_1 = (
        ~adata.obs.condition.str.contains("ctrl") & 
        (
            (adata.obs.gene_1.isin(train_conditions) & ~adata.obs.gene_2.isin(train_conditions)) | 
            (~adata.obs.gene_1.isin(train_conditions) & adata.obs.gene_2.isin(train_conditions))
        )
    )
    mask_double_perturbation_seen_2 = (
        ~adata.obs.condition.str.contains("ctrl") & 
        adata.obs.gene_1.isin(train_conditions) & 
        adata.obs.gene_2.isin(train_conditions)
    )
    adata.obs.loc[mask_single_perturbation, "subgroup"] = "single"
    adata.obs.loc[mask_double_perturbation_seen_0, "subgroup"] = "double_seen_0"
    adata.obs.loc[mask_double_perturbation_seen_1, "subgroup"] = "double_seen_1"
    adata.obs.loc[mask_double_perturbation_seen_2, "subgroup"] = "double_seen_2"

def add_embeddings(adata):
    with open('/lustre/groups/ml01/workspace/ot_perturbation/data/embeddings/gene_nargab.pkl', 'rb') as f:
        nargab_emb = pickle.load(f)
    adata.uns["nargab"] = nargab_emb

    with open('/lustre/groups/ml01/workspace/ot_perturbation/data/embeddings/gene_mix.pkl', 'rb') as f:
        mix_emb = pickle.load(f)
    adata.uns["mix"] = mix_emb


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    print(config_dict)
    split = config_dict["dataset"]["split"]
    DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/norman_new"

    adata_train_path = os.path.join(DATA_DIR, f"adata_train_pca_50_split_{split}.h5ad")
    adata_test_path = os.path.join(DATA_DIR, f"adata_test_pca_50_split_{split}.h5ad")
    adata_ood_path = os.path.join(DATA_DIR, f"adata_ood_pca_50_split_{split}.h5ad")
    
    adata_train = sc.read_h5ad(adata_train_path)
    adata_test = sc.read_h5ad(adata_test_path)
    adata_ood = sc.read_h5ad(adata_ood_path)

    add_embeddings(adata_train)
    add_embeddings(adata_test)
    add_embeddings(adata_ood)


    add_subgroup_annotations(adata_train, adata_ood)
    
    adata_ood_single = adata_ood[(adata_ood.obs["kategory"] == "ctrl") | (adata_ood.obs["subgroup"]=="single")]
    adata_ood_double_seen_0 = adata_ood[(adata_ood.obs["kategory"] == "ctrl") | (adata_ood.obs["subgroup"]=="double_seen_0")]
    adata_ood_double_seen_1 = adata_ood[(adata_ood.obs["kategory"] == "ctrl") | (adata_ood.obs["subgroup"]=="double_seen_1")]
    adata_ood_double_seen_2 = adata_ood[(adata_ood.obs["kategory"] == "ctrl") | (adata_ood.obs["subgroup"]=="double_seen_2")]
    
    cf = cellflow.model.CellFlow(adata_train, solver="otfm")
    
    # Prepare the training data and perturbation conditions
    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    perturbation_covariate_reps = dict(config_dict["dataset"]["perturbation_covariate_reps"])
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_covariate_reps,
        sample_covariates=None,
        sample_covariate_reps=None,
        split_covariates=None
    )
    
    match_fn = functools.partial(
        solver_utils.match_linear,
        epsilon=config_dict["model"]["epsilon"],
        scale_cost="mean",
        tau_a=config_dict["model"]["tau_a"],
        tau_b=config_dict["model"]["tau_b"]
    )
    
    optimizer = optax.MultiSteps(optax.adam(config_dict["model"]["learning_rate"]), config_dict["model"]["multi_steps"])
    probability_path= {config_dict["model"]["flow_type"]: config_dict["model"]["flow_noise"]}

    layers_before_pool = config_dict["model"]["layers_before_pool"]
    layers_after_pool = config_dict["model"]["layers_after_pool"]

    solver_kwargs = {"ema": config_dict["model"]["ema"]}
    time_max_period = config_dict["model"]["time_max_period"]
    time_max_period = time_max_period if time_max_period>0 else None
    
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
        time_max_period=time_max_period,
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
        adata_ood_single,
        name="ood_single",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    cf.prepare_validation_data(
        adata_ood_double_seen_0,
        name="ood_double_seen_0",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    cf.prepare_validation_data(
        adata_ood_double_seen_1,
        name="ood_double_seen_1",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    cf.prepare_validation_data(
        adata_ood_double_seen_2,
        name="ood_double_seen_2",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    
    metrics_callback = cellflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cellflow.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
    wandb_callback = cellflow.training.WandbLogger(
        project="norman_final_run", 
        out_dir="/lustre/groups/ml01/workspace/ot_perturbation/logging", 
        config=config_dict,
    )
    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]
    
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )

    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    if config_dict["training"]["save_predictions"]:

        adata_ood_ctrl = adata_ood[adata_ood.obs["control"]]
        covariate_data_test = adata_ood.obs.drop_duplicates(subset=["gene_1", "gene_2"])
        preds_test = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_test)
        all_data = []
        conditions = []

        for condition, array in preds_test.items():
            all_data.append(array)
            conditions.extend([condition] * array.shape[0])

        # Stack all data vertically to create a single array
        all_data_array = np.vstack(all_data)

        # Create a DataFrame for the .obs attribute
        obs_data = pd.DataFrame({
            'condition': conditions
        })

        # Create the Anndata object
        adata_test_result = ad.AnnData(X=np.empty((len(all_data_array), adata_train.shape[1])), obs=obs_data)
        adata_test_result.obsm["X_pca_pred"] = all_data_array
        cfpp.reconstruct_pca(query_adata=adata_test_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
        adata_save_path = os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_adata_test_with_predictions_{split}.h5ad")
        adata_test_result.write(adata_save_path)

    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)