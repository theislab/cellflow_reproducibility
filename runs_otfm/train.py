import hydra
import wandb
import sys
from omegaconf import DictConfig
import jax.numpy as jnp
from jax import random
import jax
import pathlib
import optax
import yaml
from datetime import datetime
from ott.geometry import costs
from ott.neural.flow_models.genot import GENOTLin, GENOTQuad
from ott.neural.flow_models.models import VelocityField
from ott.neural.flow_models.samplers import uniform_sampler
from ott.neural.models import base_solver
from ott.neural.models.nets import RescalingMLP
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from torch.utils.data import DataLoader
from ott.neural.data import datasets
import os

import scanpy as sc
from ot_pert.metrics import compute_metrics


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    """
    Main training function using Hydra.

    Args:
        cfg (DictConfig): Configuration parameters.

    Raises:
        Exception: Any exception during training.

    Returns:
        None
    """
    # initialize wandb
    name_tag = "genot_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    full_save_path = pathlib.Path(cfg.data.output_dir, name_tag)
    if cfg.wandb.use_wandb:
        wandb.init(
        project=cfg.wandb.project,
        config=cfg,
        name=name_tag,
        entity="ot_pert",
    )
        
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path)
    with open(pathlib.Path(full_save_path, "config.yaml"), "+w") as f:
        yaml.dump(cfg, f)

    # Load data
    adata_train = sc.read(cfg.dataset.adata_train_path)
    dss = []
    for cell_line in adata_train.obs["cell_line"].cat.categories:
        adata_cell_line = adata_train[adata_train.obs["cell_line"]==cell_line]
        source = adata_cell_line[adata_cell_line.obs["condition"]=="control"].obsm[cfg.dataset.obsm_key_train_data]
        for cond in adata_cell_line.obs["condition"].cat.categories:
            if cond == "control":
                continue
            target = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_train_data]
            conditions = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_train_data]
            dss.append(datasets.OTDataset(
                lin=source,
                target_lin=target,
                conditions=conditions
            ))
    
    train_loader = datasets.ConditionalOTDataset(dss, seed=cfg.get("seed"))

    valid_data_source = {}
    valid_data_conditions = {}
    valid_data_target = {}
    adata_valid = sc.read(cfg.dataset.adata_valid_path)
    for cell_line in adata_valid.obs["cell_line"].cat.categories:
        adata_cell_line = adata_valid[adata_valid.obs["cell_line"]==cell_line]
        source = adata_cell_line[adata_cell_line.obs["condition"]=="control"].obsm[cfg.dataset.obsm_key_valid_data]
        for cond in adata_cell_line.obs["condition"].cat.categories:
            if cond == "control":
                continue
            target = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_valid_data]
            conditions = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_valid_data]
            valid_data_source[cell_line][cond] = source
            valid_data_target[cell_line][cond] = target
            valid_data_conditions[cell_line][cond] = conditions


    test_data_source = {}
    test_data_target = {}
    test_data_conditions = {}
    adata_test = sc.read(cfg.dataset.adata_test_path)
    for cell_line in adata_test.obs["cell_line"].cat.categories:
        adata_cell_line = adata_test[adata_test.obs["cell_line"]==cell_line]
        source = adata_cell_line[adata_cell_line.obs["condition"]=="control"].obsm[cfg.dataset.obsm_key_test_data]
        for cond in adata_cell_line.obs["condition"].cat.categories:
            if cond == "control":
                continue
            target = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_test_data]
            conditions = adata_cell_line[adata_cell_line.obs["condition"]==cond].obsm[cfg.dataset.obsm_key_test_data]
            test_data_source[cell_line][cond] = source
            test_data_target[cell_line][cond] = target
            test_data_conditions[cell_line][cond] = conditions
    
    source_dim = source.shape[1]
    target_dim = source_dim
    condition_dim = conditions.shape[1]

    neural_vf = VelocityField(
        output_dim=target_dim,
        condition_dim=source_dim + condition_dim,
        latent_embed_dim=cfg.models.latent_embed_dim,
    )
    ot_solver = sinkhorn.Sinkhorn() 
    ot_matcher = base_solver.OTMatcherLinear(
        ot_solver, cost_fn=costs.SqEuclidean(), tau_a=cfg.models.tau_a, tau_b=cfg.models.tau_b
    )
    time_sampler = uniform_sampler

    optimizer = optax.adam(learning_rate=cfg.models.learning_rate)
    genot = GENOTLin(
        neural_vf,
        input_dim=source_dim,
        output_dim=target_dim,
        cond_dim=condition_dim,
        iterations=cfg.models.iteratations,
        valid_freq=cfg.models.iterations,
        ot_matcher=ot_matcher,
        unbalancedness_handler=None,
        optimizer=optimizer,
        time_sampler=time_sampler,
        k_samples_per_x=cfg.models.k_samples_per_x,
    )
    
    # Train, we don't use the valid_loader, so this can be the train_loader 
    genot(
        train_loader, train_loader
    )
    # Predict
    predicted_target_valid = jax.tree_util.tree_map(genot.transport, valid_data_source, valid_data_conditions)
    predicted_target_test = jax.tree_util.tree_map(genot.transport, test_data_source, test_data_conditions)

    # compute metrics
    valid_metrics = jax.tree_util.tree_map(compute_metrics, valid_data_target, predicted_target_valid)
    test_metrics = jax.tree_util.tree_map(compute_metrics, test_data_target, predicted_target_test)
	
    table = wandb.Table(columns=["metric", "cell_line", "condition", "value"])
    for metric in test_metrics.keys():
        for cell_line in test_metrics.keys():
            for condition in cell_line.keys():
                table.add_data(
                    metric,
                    cell_line,
                    condition,
                    test_metrics[metric][cell_line][condition],
                )
	
    wandb.log({"table": table})
    wandb.finish()
    
    
if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise