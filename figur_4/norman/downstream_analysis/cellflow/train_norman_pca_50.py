print("start imports")
import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple

import cfp
from cfp import preprocessing as cfpp
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

class SavePredictionsCallback(cfp.training.ComputationCallback):
    
    def __init__(self):
        self.log_counter = 0
        
    def on_train_begin(self) -> Any:
        self.path = wandb.run
        os.makedirs(self.path, exist_ok=True)
        print(f"Saving predictions to {self.path}")
            
        
    def on_train_end(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        return {}
    
    def on_log_iteration(
        self,
        validation_data: dict[str, dict[str, ArrayLike]],
        predicted_data: dict[str, dict[str, ArrayLike]],
    ) -> dict[str, float]:
        
        for key in validation_data.keys():
            pd.to_pickle(validation_data, os.path.join(self.path, f"{key}_validation_data_{self.log_counter}.pickle"))
            pd.to_pickle(predicted_data, os.path.join(self.path, f"{key}_predicted_data_{self.log_counter}.pickle"))
        
        self.log_counter += 1
        return {}


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    print(config_dict)
    split = config_dict["dataset"]["split"]
    adata_train_path = f"/home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata/adata_train_pca_50_split_{split}.h5ad"
    adata_test_path = f"/home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata/adata_val_pca_50_split_{split}.h5ad"
    adata_ood_path = f"/home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata/adata_test_pca_50_split_{split}.h5ad"
    adata_train = sc.read_h5ad(adata_train_path)
    adata_test = sc.read_h5ad(adata_test_path)
    adata_ood = sc.read_h5ad(adata_ood_path)
    
    cf = cfp.model.CellFlow(adata_train, solver="otfm")
    
    # Prepare the training data and perturbation conditions
    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),
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
    
    cf.prepare_validation_data(
        adata_test,
        name="test",
        n_conditions_on_log_iteration=config_dict["training"]["test_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["test_n_conditions_on_log_iteration"],
    )
    
    cf.prepare_validation_data(
        adata_ood,
        name="ood",
        n_conditions_on_log_iteration=2,#config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=2,#config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )
    
    metrics_callback = cfp.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cfp.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
    wandb_callback = cfp.training.WandbLogger(
        project="cfp_otfm_norman2", 
        out_dir="/home/haicu/soeren.becker/repos/ot_pert_reproducibility", 
        config=config_dict,
    )
    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]
    # save_preds_callback = SavePredictionsCallback()
    # callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback, save_preds_callback]
    
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )

    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    if config_dict["training"]["save_predictions"]:

        adata_ood_ctrl = adata_ood[adata_ood.obs.control.values.to_numpy(dtype=bool)]
        covariate_data_test = adata_ood.obs.drop_duplicates(subset=["gene_1", "gene_2"])
        print("predict: start.")
        preds_test = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_test)
        print("predict: done.")
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
        print("start reconstruct_pca.")
        cfpp.reconstruct_pca(query_adata=adata_test_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
        print("start done.")
        adata_save_path = os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_adata_test_with_predictions_{split}.h5ad")
        print(f"Saving results at: {adata_save_path}")
        adata_test_result.write(adata_save_path)

    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)