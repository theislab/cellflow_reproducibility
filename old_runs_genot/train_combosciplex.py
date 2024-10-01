import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax
import scanpy as sc
import wandb
from omegaconf import DictConfig, OmegaConf
from ott.neural import datasets
from ott.neural.methods.flows import dynamics, genot
from ott.neural.networks.layers import time_encoder
from ott.solvers import utils as solver_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from ot_pert.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
from ot_pert.nets.nets import GENOTVelocityFieldWithAttention
from ot_pert.utils import ConditionalLoader


def reconstruct_data(embedding, projection_matrix, mean_to_add):
    """Reconstructs data from projections."""
    return np.matmul(embedding, projection_matrix.T) + mean_to_add


def setup_logger(cfg):
    """Initialize and return a Weights & Biases logger."""
    wandb.login()
    return wandb.init(
        project=cfg.dataset.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir="/home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/bash_scripts",
        settings=wandb.Settings(start_method="thread"),
    )


def load_data(adata, cfg, *, return_dl: bool):
    """Loads data and preprocesses it based on configuration."""
    dls = []
    data_source = {}
    data_target = {}
    data_source_decoded = {}
    data_target_decoded = {}
    data_conditions = {}
    source = adata[adata.obs["condition"] == "control"].obsm[cfg.dataset.obsm_key_data]
    source_decoded = adata[adata.obs["condition"] == "control"].X

    for cond in adata.obs["condition"].cat.categories:
        if cond != "control":
            target = adata[adata.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_data]
            target_decoded = adata[adata.obs["condition"] == cond].X.A
            condition_1 = adata[adata.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_1]
            condition_2 = adata[adata.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond_2]
            assert np.all(np.all(condition_1 == condition_1[0], axis=1))
            assert np.all(np.all(condition_2 == condition_2[0], axis=1))
            expanded_arr = np.expand_dims(
                np.concatenate((condition_1[0, :][None, :], condition_2[0, :][None, :]), axis=0), axis=0
            )
            conds = np.tile(expanded_arr, (len(source), 1, 1))

            if return_dl:
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
            else:
                data_source[cond] = source
                data_target[cond] = target
                data_source_decoded[cond] = source_decoded
                data_target_decoded[cond] = target_decoded
                data_conditions[cond] = conds
    if return_dl:
        return ConditionalLoader(dls, seed=0)

    deg_dict = {k: v for k, v in adata.uns["rank_genes_groups_cov_all"].items() if k in data_conditions.keys()}

    return {
        "source": data_source,
        "target": data_target,
        "source_decoded": data_source_decoded,
        "target_decoded": data_target_decoded,
        "conditions": data_conditions,
        "deg_dict": deg_dict,
    }


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


def get_mask(x, y, var_names):
    return x[:, [gene in y for gene in var_names]]


def prepare_data(
    batch: Dict[str, jnp.ndarray],
) -> Tuple[
    Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]],
]:
    src_lin, src_quad = batch.get("src_lin"), batch.get("src_quad")
    tgt_lin, tgt_quad = batch.get("tgt_lin"), batch.get("tgt_quad")

    if src_quad is None and tgt_quad is None:  # lin
        src, tgt = src_lin, tgt_lin
        arrs = src_lin, tgt_lin
    elif src_lin is None and tgt_lin is None:  # quad
        src, tgt = src_quad, tgt_quad
        arrs = src_quad, tgt_quad
    elif all(arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)):  # fused quad
        src = jnp.concatenate([src_lin, src_quad], axis=1)
        tgt = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
        arrs = src_quad, tgt_quad, src_lin, tgt_lin
    else:
        raise RuntimeError("Cannot infer OT problem type from data.")

    return (src, batch.get("src_condition"), tgt), arrs


@hydra.main(config_path="conf", config_name="train")
def run(cfg: DictConfig):
    """Main function to handle model training and evaluation."""
    logger = setup_logger(cfg)
    adata_train = sc.read_h5ad(cfg.dataset.adata_train_path)
    adata_test = sc.read_h5ad(cfg.dataset.adata_test_path)
    adata_ood = sc.read_h5ad(cfg.dataset.adata_ood_path)
    train_data = load_data(adata_train, cfg, return_dl=False) if cfg.training.n_train_samples != 0 else {}
    test_data = load_data(adata_test, cfg, return_dl=False) if cfg.training.n_test_samples != 0 else {}
    ood_data = load_data(adata_ood, cfg, return_dl=False) if cfg.training.n_ood_samples != 0 else {}
    dl = load_data(adata_train, cfg, return_dl=True)
    comp_metrics_fn = compute_metrics_fast if cfg.training.fast_metrics else compute_metrics

    reconstruct_data_fn = functools.partial(
        reconstruct_data, projection_matrix=adata_train.varm["PCs"], mean_to_add=adata_train.varm["X_train_mean"].T
    )
    mask_fn = functools.partial(get_mask, var_names=adata_train.var_names)

    batch = next(dl)
    source_dim = batch["tgt_lin"].shape[1]
    target_dim = source_dim
    condition_dim = batch["src_condition"].shape[-1]

    vf = GENOTVelocityFieldWithAttention(
        split_dim=source_dim,
        num_heads=cfg.model.num_heads,
        qkv_feature_dim=cfg.model.qkv_feature_dim,
        max_seq_length=cfg.model.max_seq_length,
        hidden_dims=cfg.model.hidden_dims,
        time_dims=cfg.model.time_dims,
        output_dims=cfg.model.output_dims + [target_dim],
        condition_dims=cfg.model.condition_dims,
        condition_dims_forward=cfg.model.condition_dims_forward,
        condition_dims_post_attention=cfg.model.condition_dims_post_attention,
        dropout_rate=cfg.model.dropout_rate,
        time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=cfg.model.time_n_freqs),
    )

    model = genot.GENOT(
        vf,
        flow=dynamics.ConstantNoiseFlow(cfg.model.flow_noise),
        data_match_fn=jax.jit(
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
        source_dim=source_dim,
        target_dim=target_dim,
        condition_dim=condition_dim,
        rng=jax.random.PRNGKey(13),
        optimizer=optax.MultiSteps(optax.adam(cfg.model.learning_rate), cfg.model.multi_steps),
    )

    training_logs = {"loss": []}
    rng = jax.random.PRNGKey(0)

    for it in tqdm(range(cfg.training.num_iterations)):
        batch = next(iter(dl))
        batch = jtu.tree_map(jnp.asarray, batch)
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        batch = jtu.tree_map(jnp.asarray, batch)
        (src, src_cond, tgt), matching_data = prepare_data(batch)

        n = src.shape[0]
        time = model.time_sampler(rng_time, n * model.n_samples_per_src)
        latent = model.latent_noise_fn(rng_noise, (n, model.n_samples_per_src))

        tmat = model.data_match_fn(*matching_data)  # (n, m)
        src_ixs, tgt_ixs = solver_utils.sample_conditional(  # (n, k), (m, k)
            rng_resample,
            tmat,
            k=model.n_samples_per_src,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]  # (n, k, ...),  # (m, k, ...)
        if src_cond is not None:
            src_cond = src_cond[src_ixs]

        if model.latent_match_fn is not None:
            src, src_cond, tgt = model._match_latent(rng, src, src_cond, latent, tgt)

        src = src.reshape(-1, *src.shape[2:])  # (n * k, ...)
        tgt = tgt.reshape(-1, *tgt.shape[2:])  # (m * k, ...)
        latent = latent.reshape(-1, *latent.shape[2:])
        if src_cond is not None:
            src_cond = src_cond.reshape(-1, *src_cond.shape[2:])

        src = jnp.tile(jnp.expand_dims(src, 1), (1, 2, 1))

        loss, model.vf_state = model.step_fn(rng_step_fn, model.vf_state, time, src, tgt, latent, src_cond)

        training_logs["loss"].append(float(loss))
        if (it % cfg.training.valid_freq == 0) and (it > 0):
            train_loss = np.mean(training_logs["loss"][-cfg.training.valid_freq :])
            log_metrics = {"train_loss": train_loss}
            eval_step(
                cfg,
                model,
                {"train": train_data, "test": test_data, "ood": ood_data},
                log_metrics,
                reconstruct_data_fn,
                comp_metrics_fn,
                mask_fn,
            )

    if cfg.training.save_model:
        import pickle
        with open(f"{cfg.training.save_path}/{wandb.run.name}_model.pkl", 'wb') as f:
            pickle.dump(model.vf_state.params, f)
    return 1.0


def eval_step(cfg, model, data, log_metrics, reconstruct_data_fn, comp_metrics_fn, mask_fn):
    for k, dat in data.items():
        if k == "test":
            n_samples = cfg.training.n_test_samples
        if k == "train":
            n_samples = cfg.training.n_train_samples
        if k == "ood":
            n_samples = cfg.training.n_ood_samples

        if n_samples != 0:
            if n_samples > 0:
                idcs = np.random.choice(list(list(dat.values())[0]), n_samples)
                dat_source = {k: v for k, v in dat["source"].items() if k in idcs}
                dat_target = {k: v for k, v in dat["target"].items() if k in idcs}
                dat_conditions = {k: v for k, v in dat["conditions"].items() if k in idcs}
                dat_deg_dict = {k: v for k, v in dat["deg_dict"].items() if k in idcs}
                dat_target_decoded = {k: v for k, v in dat["target_decoded"].items() if k in idcs}
            else:
                dat_source = dat["source"]
                dat_target = dat["target"]
                dat_conditions = dat["conditions"]
                dat_deg_dict = dat["deg_dict"]
                dat_target_decoded = dat["target_decoded"]

            dat_source = jax.tree_util.tree_map(
                lambda x: jnp.tile(jnp.expand_dims(x, 1), (1, cfg.model.max_seq_length, 1)), dat_source
            )
            prediction = jtu.tree_map(model.transport, dat_source, dat_conditions)
            metrics = jtu.tree_map(comp_metrics_fn, dat_target, prediction)
            mean_metrics = compute_mean_metrics(metrics, prefix=f"{k}_")
            log_metrics.update(mean_metrics)

            prediction_decoded = jtu.tree_map(reconstruct_data_fn, prediction)
            metrics_decoded = jtu.tree_map(comp_metrics_fn, dat_target_decoded, prediction_decoded)
            mean_metrics_decoded = compute_mean_metrics(metrics_decoded, prefix=f"decoded_{k}_")
            log_metrics.update(mean_metrics_decoded)

            prediction_decoded_deg = jtu.tree_map(mask_fn, prediction_decoded, dat_deg_dict)
            target_decoded_deg = jax.tree_util.tree_map(mask_fn, dat_target_decoded, dat_deg_dict)
            metrics_deg = jtu.tree_map(comp_metrics_fn, target_decoded_deg, prediction_decoded_deg)
            mean_metrics_deg = compute_mean_metrics(metrics_deg, prefix=f"deg_{k}_")
            log_metrics.update(mean_metrics_deg)

    wandb.log(log_metrics)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
