import functools
import os
import sys
import traceback
from typing import Literal, Optional

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
from ott.neural.methods.flows import dynamics, otfm
from ott.neural.networks.layers import time_encoder
from ott.solvers import utils as solver_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from ot_pert.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast
from ot_pert.nets.nets import CondVelocityField
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
    for cond in adata.obs["condition"].cat.categories:
        if "Vehicle" not in cond:
            src_str_unique = list(adata[adata.obs["condition"] == cond].obs["cell_type"].unique())
            assert len(src_str_unique) == 1
            src_str = src_str_unique[0] + "_Vehicle_0.0"
            source = adata[adata.obs["condition"] == src_str].obsm[cfg.dataset.obsm_key_data]
            source_decoded = adata[adata.obs["condition"] == src_str].X.A
            target = adata[adata.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_data]
            target_decoded = adata[adata.obs["condition"] == cond].X.A
            conds = adata[adata.obs["condition"] == cond].obsm[cfg.dataset.obsm_key_cond]
            assert np.all(np.all(conds == conds[0], axis=1))
            conds = np.tile(conds[0], (len(source), 1))
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


@hydra.main(config_path="conf", config_name="train")
def run(cfg: DictConfig):
    """Main function to handle model training and evaluation."""
    logger = setup_logger(cfg)
    adata_train = sc.read_h5ad(cfg.dataset.adata_train_path)
    adata_test = sc.read_h5ad(cfg.dataset.adata_test_path)
    adata_ood = sc.read_h5ad(cfg.dataset.adata_ood_path)
    dl = load_data(adata_train, cfg, return_dl=True)
    train_data = load_data(adata_train, cfg, return_dl=False) if cfg.training.n_train_samples != 0 else {}
    test_data = load_data(adata_test, cfg, return_dl=False) if cfg.training.n_test_samples != 0 else {}
    ood_data = load_data(adata_ood, cfg, return_dl=False) if cfg.training.n_ood_samples != 0 else {}
    comp_metrics_fn = compute_metrics_fast if cfg.training.fast_metrics else compute_metrics

    reconstruct_data_fn = functools.partial(
        reconstruct_data, projection_matrix=adata_train.varm["PCs"], mean_to_add=adata_train.varm["X_train_mean"].T
    )
    mask_fn = functools.partial(get_mask, var_names=adata_train.var_names)

    batch = next(dl)
    output_dim = batch["tgt_lin"].shape[1]
    condition_dim = batch["src_condition"].shape[1]

    vf = CondVelocityField(
        hidden_dims=cfg.model.hidden_dims,
        time_dims=cfg.model.time_dims,
        output_dims=cfg.model.output_dims + [output_dim],
        condition_dims=cfg.model.condition_dims,
        dropout_rate=cfg.model.dropout_rate,
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
        optimizer=optax.MultiSteps(optax.adam(cfg.model.learning_rate), cfg.model.multi_steps),
    )

    training_logs = {"loss": []}
    rng = jax.random.PRNGKey(0)
    print("starting trainng")
    for it in tqdm(range(cfg.training.num_iterations)):
        rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
        batch = next(dl)
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
        if ((it-1) % cfg.training.valid_freq == 0) and (it > 1):
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
