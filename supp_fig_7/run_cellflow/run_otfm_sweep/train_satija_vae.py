import os, sys
import cfp
import scanpy as sc
import numpy as np
import optax
import hydra 
from omegaconf import DictConfig, OmegaConf
from functools import partial
from cfp.training import Metrics, PCADecodedMetrics
from cfp.training._callbacks import WandbLogger
from ott.solvers import utils as solver_utils
import datetime
import traceback
from typing import Dict, Literal, Optional, Tuple, NamedTuple, Any
from cfp.external import CFJaxSCVI

@hydra.main(config_path="conf", config_name="train_ifng")
def run(cfg: DictConfig):
    data_path = cfg.dataset.dataset_path
    file_suffix = cfg.dataset.file_suffix
    dataset_path = {
        "train": os.path.join(data_path, f"adata_train{file_suffix}.h5ad"),
        "test": os.path.join(data_path, f"adata_test{file_suffix}.h5ad"),
        "ood": os.path.join(data_path, f"adata_ood{file_suffix}.h5ad")
    }
    adata_train = sc.read_h5ad(dataset_path["train"])
    adata_test = sc.read_h5ad(dataset_path["test"])
    adata_ood = sc.read_h5ad(dataset_path["ood"])
    print("------------------ Data loaded ------------------")

    vae = CFJaxSCVI.load(cfg.dataset.vae_path, adata=adata_train)
    adata_train.obsm["X_scVI"] = vae.get_latent_representation(adata_train)
    adata_test.obsm["X_scVI"] = vae.get_latent_representation(adata_test)
    adata_ood.obsm["X_scVI"] = vae.get_latent_representation(adata_ood)
    
    optimizer = optax.MultiSteps(optax.adam(learning_rate=cfg.training.learning_rate), cfg.training.multi_steps)

    match_kwargs={"epsilon": cfg.model.match_kwargs.epsilon, "tau_a": cfg.model.match_kwargs.tau_a, "tau_b": cfg.model.match_kwargs.tau_b}
    match_fn = partial(solver_utils.match_linear, **match_kwargs)
    
    cf = cfp.model.CellFlow(adata_train, solver='otfm')
    cf.prepare_data(
        sample_rep="X_scVI",
        control_key="control",
        split_covariates=[
            'cell_type',
            'pathway'
        ],
        sample_covariates=[
            'pathway',
            'cell_type'
        ],
        sample_covariate_reps={
            "cell_type": "cell_type_emb",
            "pathway": "pathway_emb",
        },  
        perturbation_covariates={'gene': ('gene',)},
        perturbation_covariate_reps={
            "gene": "gene_emb",
        }, 
    )

    cf.prepare_validation_data(
        adata_test,
        name=cfg.training.test_name,
        n_conditions_on_log_iteration=cfg.training.test_n_conditions.iteration,
        n_conditions_on_train_end=cfg.training.test_n_conditions.end
    )
    cf.prepare_validation_data(
        adata_ood,
        name=cfg.training.ood_name,
        n_conditions_on_log_iteration=cfg.training.ood_n_conditions.iteration,
        n_conditions_on_train_end=cfg.training.ood_n_conditions.end
    )

    cf.prepare_model(
        hidden_dims=cfg.model.hidden_dims,
        decoder_dims=cfg.model.decoder_dims,
        condition_embedding_dim=cfg.model.condition_embedding_dim,
        time_encoder_dims=cfg.model.time_encoder_dims,
        flow={"constant_noise": cfg.model.flow.constant_noise},
        hidden_dropout=cfg.model.hidden_dropout,
        decoder_dropout=cfg.model.decoder_dropout,
        match_fn=match_fn,
        optimizer=optimizer,
        layers_after_pool={
            "layer_type": cfg.model.layers_after_pool.layer_type,
            "dims": cfg.model.layers_after_pool.dims
        }
    )
    print("------------------ Model prepared ------------------")

    callbacks = []
    metrics_callback = Metrics(metrics=cfg.logger.metrics)
    callbacks.append(metrics_callback)

    if cfg.logger.calc_decoded_metrics:
        decoded_metrics_callback = cfp.training.VAEDecodedMetrics(vae, adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
        callbacks.append(decoded_metrics_callback)

    wandb_callback = WandbLogger(project=cfg.logger.project, out_dir=cfg.logger.out_dir, config=OmegaConf.to_container(cfg, resolve=True))
    callbacks.append(wandb_callback)
    print("------------------ Callbacks prepared ------------------")

    cf.train(
        num_iterations=cfg.training.num_iterations, 
        callbacks=callbacks, 
        valid_freq=cfg.training.valid_freq, 
        batch_size=cfg.training.batch_size,
        #checkpoint_dir=cfg.logger.save_model_path
    )

    print("------------------ Training done ------------------")

    if cfg.logger.save_model:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(os.path.join(cfg.logger.save_model_path,timestamp))
        cf.save(dir_path=os.path.join(cfg.logger.save_model_path,timestamp), file_prefix=cfg.logger.project)
        print("------------------ Model saved ------------------")

    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
        