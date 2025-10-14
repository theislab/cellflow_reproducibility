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
from cfp.metrics import compute_r_squared, compute_e_distance
from cfp.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
import gc
import pickle


def compute_metrics(adata_pred: ad.AnnData, adata_ood_true: ad.AnnData) -> dict:
    dict_to_log = {}

    # standard metrics
    ood_r_squared = compute_r_squared(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    
    decoded_ood_r_squared = compute_r_squared(adata_ood_true.X.toarray(), adata_pred.layers["X_recon"])
    
    # metrics to return
    dict_to_log["ood_r_squared"] = ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    dict_to_log["decoded_ood_r_squared"] = decoded_ood_r_squared
    return dict_to_log




@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    donor_held_out = config_dict["dataset"]["donor_held_out"]
    idx_given_donor = str(config_dict["dataset"]["idx_given_donor"])
    control_key = "is_control"

    adata_train = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{donor_held_out}/{str(idx_given_donor)}/adata_train_{donor_held_out}.h5ad")
    adata_ood_perturbed  = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_donor/{donor_held_out}/{str(idx_given_donor)}/adata_ood_{donor_held_out}.h5ad")

    cytokines_to_impute = adata_train.uns["split_info"][idx_given_donor]["cytokines_to_impute"]
    cytokines_to_train_data = adata_train.uns["split_info"][idx_given_donor]["cytokines_to_train_data"]
    cfp.preprocessing.centered_pca(adata_train, n_comps=100, keep_centered_data=False)
    cfp.preprocessing.project_pca(query_adata=adata_ood_perturbed, ref_adata=adata_train)
    
    adata_ctrl = adata_train[adata_train.obs[control_key].to_numpy()]
    
    adata_ctrl_subsetted = []
    for donor in adata_ctrl.obs["donor"].unique():
        adata_tmp = adata_ctrl[adata_ctrl.obs["donor"]==donor]
        if adata_tmp.n_obs > 10000:
            sc.pp.subsample(adata_tmp, n_obs=10000)
        adata_ctrl_subsetted.append(adata_tmp)
    adata_ctrl = ad.concat(adata_ctrl_subsetted)

    
    adata_ctrl.uns = adata_train.uns.copy()
    adata_ood_perturbed.uns = adata_train.uns.copy()

    if len(cytokines_to_train_data) == 1: # if we don't train on held out donor at all, remove its PBS
        adata_ctrl_donor = adata_train[adata_train.obs["donor"]==donor_held_out]
        assert len(adata_ctrl_donor.obs["cytokine"].unique()) ==1
        adata_train = adata_train[adata_train.obs["donor"]!=donor_held_out]
        
    cf = cfp.model.CellFlow(adata_train, solver="otfm")

    perturbation_covariates = {"cytokines": ["cytokine"]}
    split_covariates = ["donor"]
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key=control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps={"cytokines": "esm2_embeddings"},
        sample_covariates=("donor",),
        sample_covariate_reps={"donor": "donor_embeddings"},
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

    adata_ood = ad.concat((adata_ctrl, adata_ood_perturbed))
    adata_ood.uns = adata_train.uns.copy()
    
    metrics_callback = cfp.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cfp.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared"])
    wandb_callback = cfp.training.WandbLogger(project="cfp_pbmc_new_donor", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]
    
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    covariate_data = adata_ood_perturbed.obs.drop_duplicates(subset=["condition"])
    if len(cytokines_to_train_data) > 1: 
        adata_ctrl_donor = adata_train[(adata_train.obs['donor']==donor_held_out) & (adata_train.obs["cytokine"]=="PBS")]
    if adata_ctrl_donor.n_obs > 10000:
        sc.pp.subsample(adata_ctrl_donor, n_obs=10000)
    
    for i in range(len(covariate_data)):
        cov_data_tmp = covariate_data.iloc[[i]]
        preds = cf.predict(adata=adata_ctrl_donor, sample_rep="X_pca", condition_id_key="condition", covariate_data=cov_data_tmp)

    
        for condition, array in preds.items():
            all_data = []
            conditions = []

            all_data.append(array)
            conditions.extend([condition] * array.shape[0])

            # Stack all data vertically to create a single array
            all_data_array = np.vstack(all_data)

            # Create a DataFrame for the .obs attribute
            obs_data = pd.DataFrame({
                'condition': conditions
            })

            # Create the Anndata object
            adata_pred = ad.AnnData(X=np.empty((len(all_data_array),adata_train.n_vars)), obs=obs_data)
            adata_pred.obsm["X_pca"] = all_data_array
            adata_pred.uns["cytokine_in_train"] = cytokines_to_train_data
            cfp.preprocessing.reconstruct_pca(query_adata=adata_pred, ref_adata=adata_train, use_rep="X_pca", layers_key_added = "X_recon")
            cond_orig = condition
            condition = condition + "_" + str(len(adata_pred.uns["cytokine_in_train"]))
            adata_pred.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}_preds.h5ad"))
            out = compute_metrics(adata_pred=adata_pred, adata_ood_true=adata_ood_perturbed[adata_ood_perturbed.obs["condition"]==cond_orig])
            pd.DataFrame.from_dict(out, columns=[condition], orient="index").to_csv(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}.csv"))
            wandb.log({condition: out})
        
        
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
