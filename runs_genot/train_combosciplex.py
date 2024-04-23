import functools
import os
import sys
from typing import Literal, Optional

import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pandas as pd
import scanpy as sc
import wandb
from omegaconf import DictConfig, OmegaConf
from ott.neural import datasets
from ott.neural.methods.flows import dynamics, otfm
from ott.neural.networks.layers import time_encoder
from ott.solvers import utils as solver_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from ot_pert.metrics import compute_mean_metrics, compute_metrics
from ot_pert.utils import ConditionalLoader


def reconstruct_data(embedding: np.ndarray, projection_matrix: np.ndarray, mean_to_add: np.ndarray) -> np.ndarray:
    return np.matmul(embedding, projection_matrix.T) + mean_to_add


@hydra.main(config_path="conf", config_name="train", version_base=None)
def run(cfg: DictConfig):
    """
    Main training function using Hydra.

    Args:
        cfg (DictConfig): Configuration parameters.

    Raises
    ------
        Exception: Any exception during training.

    Returns
    -------
        None
    """
    experiment_logger = setup_logger(cfg=cfg)

    def data_match_fn(
        src_lin: Optional[jnp.ndarray],
        tgt_lin: Optional[jnp.ndarray],
        src_quad: Optional[jnp.ndarray],
        tgt_quad: Optional[jnp.ndarray],
        *,
        typ: Literal["lin", "quad", "fused"],
        epsilon: float = 1e-2,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
    ) -> jnp.ndarray:
        if typ == "lin":
            return solver_utils.match_linear(
                x=src_lin, y=tgt_lin, scale_cost="mean", epsilon=epsilon, tau_a=tau_a, tau_b=tau_b
            )
        if typ == "quad":
            return solver_utils.match_quadratic(xx=src_quad, yy=tgt_quad)
        if typ == "fused":
            return solver_utils.match_quadratic(xx=src_quad, yy=tgt_quad, x=src_lin, y=tgt_lin)
        raise NotImplementedError(f"Unknown type: {typ}.")

    # Load data
    adata_train = sc.read(cfg.dataset.adata_train_path)
    dls = []

    train_data_source = {}
    train_data_target = {}
    train_data_source_decoded = {}
    train_data_target_decoded = {}
    train_data_conditions = {}

    source = adata_train[adata_train.obs["condition"] == "control"].obsm[cfg.dataset.obsm_key_data]
    source_decoded = adata_train[adata_train.obs["condition"] == "control"].X
    for cond in adata_train.obs["condition"].cat.categories:
        if cond == "control":
            continue
        target = adata_train[adata_train.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_data]
        target_decoded = adata_train[adata_train.obs["condition"] == cond].X.A
        condition_1 = adata_train[adata_train.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_1]
        condition_2 = adata_train[adata_train.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_2]
        assert np.all(np.all(condition_1 == condition_1[0], axis=1))
        assert np.all(np.all(condition_2 == condition_2[0], axis=1))
        expanded_arr = np.expand_dims(
            np.concatenate((condition_1[0, :][None, :], condition_2[0, :][None, :]), axis=0), axis=0
        )
        conds = np.tile(expanded_arr, (len(source), 1, 1))
        dls.append(
            DataLoader(
                datasets.OTDataset(
                    datasets.OTData(
                        lin=source,
                        condition=conds,
                    ),
                    datasets.OTData(lin=target),
                ),
                batch_size=cfg.training.batch_size,
                shuffle=True,
            )
        )
        train_data_source[cond] = source
        train_data_target[cond] = target
        train_data_conditions[cond] = conds
        train_data_source_decoded[cond] = source_decoded
        train_data_target_decoded[cond] = target_decoded

    train_loader = ConditionalLoader(dls, seed=0)

    test_data_source = {}
    test_data_target = {}
    test_data_source_decoded = {}
    test_data_target_decoded = {}
    test_data_conditions = {}
    adata_test = sc.read(cfg.dataset.adata_test_path)
    source = adata_test[adata_test.obs["condition"] == "control"].obsm[cfg.dataset.obsm_key_data]
    source_decoded = adata_test[adata_test.obs["condition"] == "control"].X
    for cond in adata_test.obs["condition"].cat.categories:
        if cond == "control":
            continue
        target = adata_test[adata_test.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_data]
        target_decoded = adata_test[adata_test.obs["condition"] == cond].X.A
        condition_1 = adata_test[adata_test.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_1]
        condition_2 = adata_test[adata_test.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_2]
        assert np.all(np.all(condition_1 == condition_1[0], axis=1))
        assert np.all(np.all(condition_2 == condition_2[0], axis=1))
        expanded_arr = np.expand_dims(
            np.concatenate((condition_1[0, :][None, :], condition_2[0, :][None, :]), axis=0), axis=0
        )
        conds = np.tile(expanded_arr, (len(source), 1, 1))
        test_data_source[cond] = source
        test_data_target[cond] = target
        test_data_source_decoded[cond] = source_decoded
        test_data_target_decoded[cond] = target_decoded
        test_data_conditions[cond] = conds

    ood_data_source = {}
    ood_data_target = {}
    ood_data_source_decoded = {}
    ood_data_target_decoded = {}
    ood_data_conditions = {}
    adata_ood = sc.read(cfg.dataset.adata_ood_path)
    source = adata_ood[adata_ood.obs["condition"] == "control"].obsm[cfg.dataset.obsm_key_data]
    source_decoded = adata_ood[adata_ood.obs["condition"] == "control"].X
    for cond in adata_ood.obs["condition"].cat.categories:
        if cond == "control":
            continue
        target = adata_ood[adata_ood.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_data]
        target_decoded = adata_ood[adata_ood.obs["condition"] == cond].X.A
        condition_1 = adata_ood[adata_ood.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_1]
        condition_2 = adata_ood[adata_ood.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_2]
        assert np.all(np.all(condition_1 == condition_1[0], axis=1))
        assert np.all(np.all(condition_2 == condition_2[0], axis=1))
        expanded_arr = np.expand_dims(
            np.concatenate((condition_1[0, :][None, :], condition_2[0, :][None, :]), axis=0), axis=0
        )
        conds = np.tile(expanded_arr, (len(source), 1, 1))
        ood_data_source[cond] = source
        ood_data_target[cond] = target
        ood_data_source_decoded[cond] = source_decoded
        ood_data_target_decoded[cond] = target_decoded
        ood_data_conditions[cond] = conds

    reconstruct_data_fn = functools.partial(
        reconstruct_data, projection_matrix=adata_train.varm["PCs"], mean_to_add=adata_train.varm["X_train_mean"].T
    )

    train_deg_dict = {
        k: v for k, v in adata_train.uns["rank_genes_groups_cov_all"].items() if k in train_data_conditions.keys()
    }
    test_deg_dict = {
        k: v for k, v in adata_train.uns["rank_genes_groups_cov_all"].items() if k in test_data_conditions.keys()
    }
    ood_deg_dict = {
        k: v for k, v in adata_train.uns["rank_genes_groups_cov_all"].items() if k in ood_data_conditions.keys()
    }

    def get_mask(x, y):
        return x[:, [gene in y for gene in adata_train.var_names]]

    source_dim = source.shape[1]
    target_dim = source_dim
    condition_dim = condition_1.shape[1]

    source_dim = source.shape[1]
    target_dim = source_dim
    condition_dim = condition_1.shape[1]

    print(cfg)

    vf = VelocityFieldWithAttention(
        num_heads=cfg.model.num_heads,
        qkv_feature_dim=cfg.model.qkv_feature_dim,
        max_seq_length=cfg.model.max_seq_length,
        hidden_dims=cfg.model.hidden_dims,
        time_dims=cfg.model.time_dims,
        output_dims=cfg.model.output_dims + [target_dim],
        condition_dims=cfg.model.condition_dims,
        time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=cfg.model.time_n_freqs),
    )

    model = otfm.OTFlowMatching(
        vf,
        flow=dynamics.ConstantNoiseFlow(cfg.model.flow_noise),
        match_fn=jax.jit(
            functools.partial(
                data_match_fn,
                typ="lin",
                src_quad=None,
                tgt_quad=None,
                epsilon=cfg.model.epsilon,
                tau_a=cfg.model.tau_a,
                tau_b=cfg.model.tau_b,
            )
        ),
        condition_dim=condition_dim,
        rng=jax.random.PRNGKey(13),
        optimizer=optax.MultiSteps(optax.adam(learning_rate=cfg.model.learning_rate), cfg.model.multi_steps),
    )

    training_logs = {"loss": []}

    rng = jax.random.PRNGKey(0)
    for it in tqdm(range(cfg.training.num_iterations)):
        rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
        batch = next(train_loader)
        batch = jtu.tree_map(jnp.asarray, batch)

        src, tgt = batch["src_lin"], batch["tgt_lin"]
        src_cond = batch.get("src_condition")

        if model.match_fn is not None:
            tmat = model.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]
            src_cond = None if src_cond is None else src_cond[src_ixs]

        model.vf_state, loss = model.step_fn(
            rng_step_fn,
            model.vf_state,
            src,
            tgt,
            src_cond,
        )

        training_logs["loss"].append(float(loss))
        if (it % cfg.training.valid_freq == 0) and (it > 0):
            valid_losses = []
            for cond in test_data_source.keys():
                src = test_data_source[cond]
                tgt = test_data_target[cond]
                src_cond = test_data_conditions[cond]
                if model.match_fn is not None:
                    tmat = model.match_fn(src, tgt)
                    src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
                    src, tgt = src[src_ixs], tgt[tgt_ixs]
                    src_cond = None if src_cond is None else src_cond[src_ixs]
                _, valid_loss = model.step_fn(
                    rng,
                    model.vf_state,
                    src,
                    tgt,
                    src_cond,
                )
                valid_losses.append(valid_loss)

            # predicted_target_train = jax.tree_util.tree_map(model.transport, train_data_source, train_data_conditions)
            # train_metrics = jax.tree_util.tree_map(compute_metrics, train_data_target, predicted_target_train)
            # mean_train_metrics = compute_mean_metrics(train_metrics, prefix="train_")

            # predicted_target_train_decoded = jax.tree_util.tree_map(reconstruct_data_fn, predicted_target_train)
            # train_metrics_decoded = jax.tree_util.tree_map(compute_metrics, train_data_target_decoded, predicted_target_train_decoded)
            # mean_train_metrics_decoded = compute_mean_metrics(train_metrics_decoded, prefix="decoded_train_")

            # train_deg_target_decoded_predicted = jax.tree_util.tree_map(get_mask, predicted_target_train_decoded, train_deg_dict)
            # train_deg_target_decoded = jax.tree_util.tree_map(get_mask, train_data_target_decoded, test_deg_dict)

            predicted_target_test = jax.tree_util.tree_map(model.transport, test_data_source, test_data_conditions)
            test_metrics = jax.tree_util.tree_map(compute_metrics, test_data_target, predicted_target_test)
            mean_test_metrics = compute_mean_metrics(test_metrics, prefix="test_")

            predicted_target_test_decoded = jax.tree_util.tree_map(reconstruct_data_fn, predicted_target_test)
            test_metrics_decoded = jax.tree_util.tree_map(
                compute_metrics, test_data_target_decoded, predicted_target_test_decoded
            )
            mean_test_metrics_decoded = compute_mean_metrics(test_metrics_decoded, prefix="decoded_test_")

            test_deg_target_decoded_predicted = jax.tree_util.tree_map(
                get_mask, predicted_target_test_decoded, test_deg_dict
            )
            test_deg_target_decoded = jax.tree_util.tree_map(get_mask, test_data_target_decoded, test_deg_dict)
            deg_test_metrics_encoded = jax.tree_util.tree_map(
                compute_metrics, test_deg_target_decoded, test_deg_target_decoded_predicted
            )
            deg_mean_test_metrics_encoded = compute_mean_metrics(deg_test_metrics_encoded, prefix="deg_test_")

            predicted_target_ood = jax.tree_util.tree_map(model.transport, ood_data_source, ood_data_conditions)
            ood_metrics = jax.tree_util.tree_map(compute_metrics, ood_data_target, predicted_target_ood)
            mean_ood_metrics = compute_mean_metrics(ood_metrics, prefix="ood_")

            predicted_target_ood_decoded = jax.tree_util.tree_map(reconstruct_data_fn, predicted_target_ood)
            ood_metrics_decoded = jax.tree_util.tree_map(
                compute_metrics, ood_data_target_decoded, predicted_target_ood_decoded
            )
            mean_ood_metrics_decoded = compute_mean_metrics(ood_metrics_decoded, prefix="decoded_ood_")

            ood_deg_target_decoded_predicted = jax.tree_util.tree_map(
                get_mask, predicted_target_ood_decoded, ood_deg_dict
            )
            ood_deg_target_decoded = jax.tree_util.tree_map(get_mask, ood_data_target_decoded, ood_deg_dict)
            deg_ood_metrics_encoded = jax.tree_util.tree_map(
                compute_metrics, ood_deg_target_decoded, ood_deg_target_decoded_predicted
            )
            deg_mean_ood_metrics_encoded = compute_mean_metrics(deg_ood_metrics_encoded, prefix="deg_ood_")

            loss_dict = {
                "train_loss": np.mean(training_logs["loss"][-cfg.training.valid_freq :]),
                "valid_loss": np.mean(valid_losses),
            }
            loss_dict.update(mean_test_metrics)
            loss_dict.update(mean_ood_metrics)
            # loss_dict.update(mean_train_metrics)
            # loss_dict.update(mean_train_metrics_decoded)
            loss_dict.update(mean_test_metrics_decoded)
            loss_dict.update(mean_ood_metrics_decoded)
            loss_dict.update(deg_mean_ood_metrics_encoded)
            loss_dict.update(deg_mean_test_metrics_encoded)
            wandb.log(loss_dict)

    output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/"
    pd.DataFrame.from_dict(ood_metrics).to_csv(os.path.join(output_dir, "ood_metrics_encoded.csv"))
    pd.DataFrame.from_dict(ood_metrics_decoded).to_csv(os.path.join(output_dir, "ood_metrics_decoded.csv"))
    pd.DataFrame.from_dict(deg_ood_metrics_encoded).to_csv(os.path.join(output_dir, "deg_metrics_decoded.csv"))
    return mean_ood_metrics["ood_sinkhorn_div_1"]  # for hyperparameter tuning


def setup_logger(cfg):
    import wandb

    wandb.login()

    wandb.init(
        project="otfm_combosciplex",
        settings=wandb.Settings(
            start_method="thread"
        ),  # avoid hanging at start-up, see: https://docs.wandb.ai/guides/integrations/hydra
        config=OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        ),
        dir="/home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/bash_scripts",
    )
    return wandb


if __name__ == "__main__":
    import traceback

    try:
        run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
