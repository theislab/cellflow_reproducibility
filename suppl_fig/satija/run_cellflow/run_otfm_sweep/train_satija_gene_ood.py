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

@hydra.main(config_path="conf", config_name="train_ifng")
def run(cfg: DictConfig):
    # Dataset path and adata suffix
    data_path = cfg.dataset.dataset_path
    file_suffix = cfg.dataset.file_suffix
    
    # Read dataset 
    dataset_path = {
        "train": os.path.join(data_path, f"adata_train{file_suffix}.h5ad"),
        "test": os.path.join(data_path, f"adata_test{file_suffix}.h5ad"),
        "ood": os.path.join(data_path, f"adata_ood{file_suffix}.h5ad")
    }
    adata_train = sc.read_h5ad(dataset_path["train"])
    adata_test = sc.read_h5ad(dataset_path["test"])
    adata_ood = sc.read_h5ad(dataset_path["ood"])
    print("------------------ Data loaded ------------------")
    
    cf = cfp.model.CellFlow(adata_train, solver='otfm')
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates={'gene': ('gene',)},
        perturbation_covariate_reps={
            "gene": "gene_emb",
        }, 
        sample_covariates=[
            'pathway',
            'cell_type'
        ],
        sample_covariate_reps={
            "cell_type": "cell_type_emb",
            "pathway": "pathway_emb",
        },  
        split_covariates=[
            'cell_type',
            'pathway'
        ]
    )
    
    # Match kwargs
    match_kwargs={"epsilon": cfg.model.epsilon,
                  "scale_cost": "mean",
                  "tau_a": cfg.model.tau_a, 
                  "tau_b": cfg.model.tau_b}
    
    match_fn = partial(solver_utils.match_linear, **match_kwargs)
    
    # Optimizer 
    optimizer = optax.MultiSteps(optax.adam(learning_rate=cfg.training.learning_rate), cfg.training.multi_steps)
    flow = {cfg.model.flow_type: cfg.model.flow_noise}
    cf.prepare_model(
        encode_conditions=True,
        condition_embedding_dim=cfg.model.condition_embedding_dim,
        pooling=cfg.model.pooling,
        time_encoder_dims=cfg.model.time_encoder_dims,
        time_encoder_dropout=cfg.model.time_encoder_dropout,
        hidden_dims=cfg.model.hidden_dims,
        hidden_dropout=cfg.model.hidden_dropout,
        decoder_dims=cfg.model.decoder_dims,
        decoder_dropout=cfg.model.decoder_dropout,
        layers_before_pool=cfg.model.layers_before_pool,
        layers_after_pool=cfg.model.layers_after_pool,
        cond_output_dropout=cfg.model.cond_output_dropout,
        time_freqs=cfg.model.time_freqs,
        match_fn=match_fn,
        optimizer=optimizer,
        flow=flow, 
        layer_norm_before_concatenation=cfg.model.layer_norm_before_concatenation,
        linear_projection_before_concatenation=cfg.model.linear_projection_before_concatenation,
    )
    print("------------------ Model prepared ------------------")
    
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

    callbacks = []
    metrics_callback = Metrics(metrics=cfg.logger.metrics)
    callbacks.append(metrics_callback)

    if cfg.logger.calc_decoded_metrics:
        decoded_metrics_callback = PCADecodedMetrics(
            ref_adata=adata_train,
            metrics=cfg.logger.decoded_metrics
        )
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
        