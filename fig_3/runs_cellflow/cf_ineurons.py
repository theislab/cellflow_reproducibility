import sys
import os
import warnings
import yaml
from typing import Literal
from functools import partial, reduce

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad

import optax
import flax.linen as nn
from ott.solvers import utils as solver_utils

import cfp
from cfp.external._scvi import CFJaxSCVI
from utils import *


warnings.filterwarnings("ignore")


def run_otfm_ineurons(
    *,
    h5ad: str,
    train_datasets: list[Literal["glut_post", "glut_pre", "gaba_post", "gaba_pre"]],
    test_dataset: Literal["glut_post", "gaba_post"],
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
    plot_umap_combs: list[str],
    plot_umap_conds: list[str],
    plot_heatmap_combs: list[str],
    plot_heatmap_conds: list[str],
    latent_space: Literal["pca"] | str,
    n_dims_train: int | None,
    n_dims_eval: int,
    vae_eval_rep: str,
    h5ad_vae_train: str | None,
    vae_train_path: str | None,
    h5ad_vae_eval: str | None,
    vae_eval_path: str | None,
    out_dir: str,
    out_prefix: str,
    save_adata: bool,
    save_model: bool,
    minimal: bool,
    iterations: int,
    batch_size: int,
    lr_init: float,
    lr_schedule: Literal["constant", "cosine"],
    lr_b1: float,
    lr_b2: float,
    n_src_cells: int,
    n_src_cells_eval: int | None,
    opt_n_steps: int,
    cond_embed_dim: int,
    pooling: Literal["mean", "attention_token", "attention_seed"],
    pool_sample_covariates: bool,
    pool_num_heads: int,
    pool_qkv_dim: int,
    pool_dropout: float,
    pool_act_fn: Literal["relu", "silu", "leaky_relu"],
    pool_seed_dim: int,
    pool_transformer_block: bool,
    pool_layer_norm: bool,
    cond_layers_before_pool: Literal["mlp", "self_attention"],
    cond_n_layers_before_pool: int,
    cond_hidden_dim_before_pool: int,
    cond_dropout_before_pool: float,
    cond_layers_after_pool: Literal["mlp", "self_attention"],
    cond_n_layers_after_pool: int,
    cond_hidden_dim_after_pool: int,
    cond_dropout_after_pool: float,
    cond_output_dropout: float,
    cond_num_heads: int,
    cond_transformer: bool,
    cond_layer_norm: bool,
    cond_act_fn: Literal["relu", "silu", "leaky_relu"],
    ot_num_layers_t: int,
    ot_num_layers_x: int,
    ot_num_layers_decoder: int,
    ot_t_hidden_dim: int,
    ot_x_hidden_dim: int,
    ot_t_embed_dim: int,
    ot_x_embed_dim: int,
    ot_joint_hidden_dim: int,
    ot_act_fn: Literal["relu", "silu", "leaky_relu"],
    ot_dropout_t: float,
    ot_dropout_x: float,
    ot_dropout_decoder: float,
    ot_flow_type: Literal["constant_noise", "bridge"],
    ot_flow_noise: float,
    ot_eps: float,
    ot_tau_a: float,
    ot_tau_b: float,
    ot_n_frequencies: int,
) -> None:

    ####################
    #### Load data #####
    ####################

    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    adata = sc.read_h5ad(h5ad)
    adata = adata[adata.obs["dataset"].isin(train_datasets)]

    # load VAEs
    sample_rep = "X_pca" if latent_space == "pca" else latent_space
    vae_train, vae_eval = load_vae(
        h5ad, latent_space, h5ad_vae_train, vae_train_path, h5ad_vae_eval, vae_eval_path
    )

    ####################
    ### Prepare data ###
    ####################

    # create covariate representations
    adata, condition_keys = get_covariate_reps(adata, molecules, train_datasets)

    # split data
    adata_train, adata_eval, sweep_conds = split_data(
        adata,
        test_dataset,
        train_combs,
        exclude_combs,
        eval_combs,
        sweep_combs,
        train_conds,
        eval_conds,
        sweep_conds,
    )

    # run PCA
    run_pca(adata, adata_train, adata_eval, latent_space, n_dims_train, n_dims_eval)

    # add source cells - mean of samples from training set
    adata_ctrl, adata_train_full = generate_adata_with_source(
        adata_train, molecules, sample_rep, n_src_cells
    )
    adata_train_full.uns, adata_eval.uns = adata.uns, adata.uns

    ####################
    ### Prepare train ##
    ####################

    act_fns = {"relu": nn.relu, "silu": nn.silu, "leaky_relu": nn.leaky_relu}
    cond_act_fn = act_fns[cond_act_fn]
    ot_act_fn = act_fns[ot_act_fn]

    # layers before pooling
    if cond_n_layers_before_pool == 0:
        layers_before_pool = ()
    elif cond_layers_before_pool == "mlp":
        layers_before_pool = (
            {
                "layer_type": "mlp",
                "dims": [cond_hidden_dim_before_pool] * cond_n_layers_before_pool,
                "dropout_rate": cond_dropout_before_pool,
                "act_fn": cond_act_fn,
            },
        )
    elif cond_layers_before_pool == "self_attention":
        layers_before_pool = (
            {
                "layer_type": "self_attention",
                "num_heads": [cond_num_heads] * cond_n_layers_before_pool,
                "qkv_dim": [cond_hidden_dim_before_pool] * cond_n_layers_before_pool,
                "dropout_rate": cond_dropout_before_pool,
                "transformer_block": cond_transformer,
                "layer_norm": cond_layer_norm,
                "act_fn": cond_act_fn,
            },
        )

    # pooling
    if pooling == "attention_seed":
        pool_kwargs = {
            "num_heads": pool_num_heads,
            "v_dim": pool_qkv_dim,
            "seed_dim": pool_seed_dim,
            "dropout_rate": pool_dropout,
            "transformer_block": pool_transformer_block,
            "layer_norm": pool_layer_norm,
            "act_fn": pool_act_fn,
        }
    elif pooling == "attention_token":
        pool_kwargs = {
            "num_heads": pool_num_heads,
            "qkv_dim": pool_qkv_dim,
            "dropout_rate": pool_dropout,
            "act_fn": pool_act_fn,
        }
    else:
        pool_kwargs = {}

    # layers after pooling
    if cond_n_layers_after_pool == 0:
        layers_after_pool = ()
    elif cond_layers_after_pool == "mlp":
        layers_after_pool = (
            {
                "layer_type": "mlp",
                "dims": [cond_hidden_dim_after_pool] * cond_n_layers_after_pool,
                "dropout_rate": cond_dropout_after_pool,
                "act_fn": cond_act_fn,
            },
        )
    elif cond_layers_after_pool == "self_attention":
        layers_after_pool = (
            {
                "layer_type": "self_attention",
                "num_heads": [cond_num_heads] * cond_n_layers_after_pool,
                "qkv_dim": [cond_hidden_dim_after_pool] * cond_n_layers_after_pool,
                "dropout_rate": cond_dropout_after_pool,
                "transformer_block": cond_transformer,
                "layer_norm": cond_layer_norm,
                "act_fn": cond_act_fn,
            },
        )

    if lr_schedule == "constant":
        lr_schedule = optax.constant_schedule(lr_init)
    elif lr_schedule == "cosine":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr_init,
            decay_steps=iterations,
        )
    optimizer = optax.MultiSteps(
        optax.adam(learning_rate=lr_schedule, b1=lr_b1, b2=lr_b2), opt_n_steps
    )

    ####################
    ###### Run OT ######
    ####################

    sample_covariates = []
    sample_covariate_reps = {}
    if len(train_datasets) > 1:
        sample_covariates.append("dataset")
        sample_covariate_reps["dataset"] = "dataset"
    if "glut_pre" in train_datasets or "gaba_pre" in train_datasets:
        sample_covariates.append("media")
        sample_covariate_reps["media"] = "media"

    cf = cfp.model.CellFlow(adata_train_full, solver="otfm")
    cf.prepare_data(
        sample_rep=sample_rep,
        control_key="CTRL",
        perturbation_covariates={"conditions": condition_keys},
        perturbation_covariate_reps={"conditions": "conditions"},
        sample_covariates=sample_covariates,
        sample_covariate_reps=sample_covariate_reps,
    )
    cf.prepare_model(
        encode_conditions=True,
        condition_embedding_dim=cond_embed_dim,
        time_encoder_dims=[ot_t_hidden_dim] * (ot_num_layers_t - 1) + [ot_t_embed_dim],
        time_encoder_dropout=ot_dropout_t,
        hidden_dims=[ot_x_hidden_dim] * (ot_num_layers_x - 1) + [ot_x_embed_dim],
        hidden_dropout=ot_dropout_x,
        decoder_dims=[ot_joint_hidden_dim] * ot_num_layers_decoder,
        decoder_dropout=ot_dropout_decoder,
        pooling=pooling,
        pooling_kwargs=pool_kwargs,
        pool_sample_covariates=pool_sample_covariates,
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        cond_output_dropout=cond_output_dropout,
        vf_act_fn=ot_act_fn,
        time_freqs=ot_n_frequencies,
        flow={ot_flow_type: ot_flow_noise},
        match_fn=partial(
            solver_utils.match_linear,
            epsilon=ot_eps,
            scale_cost="mean",
            tau_a=ot_tau_a,
            tau_b=ot_tau_b,
        ),
        optimizer=optimizer,
    )
    cf.train(
        num_iterations=iterations,
        batch_size=batch_size,
        valid_freq=iterations * 2,
    )

    ####################
    #### Evaluate OT ###
    ####################

    n_src_cells_eval = n_src_cells_eval or n_src_cells
    if n_src_cells_eval > n_src_cells:
        adata_ctrl, _ = generate_adata_with_source(
            adata_train, molecules, sample_rep, n_src_cells_eval
        )
    else:
        adata_ctrl = adata_ctrl[0:n_src_cells_eval]
    adata_ctrl.uns = adata.uns
    obs_pred = adata_eval.obs.drop_duplicates("condition")
    x_pred = cf.predict(
        adata_ctrl,
        sample_rep=sample_rep,
        covariate_data=obs_pred,
        condition_id_key="condition",
    )

    # obtain predictions
    adatas_pred = []
    for condition in x_pred.keys():
        adata_pred = ad.AnnData(X=adata_ctrl.X)
        adata_pred.obs["dataset"] = "glut_post"
        adata_pred.obs["comb"] = adata_eval.obs["comb"][
            adata_eval.obs["condition"] == condition
        ].values[0]
        adata_pred.obs["condition"] = condition
        adata_pred.obsm[sample_rep] = x_pred[condition]
        adatas_pred.append(adata_pred)
    adata_pred = ad.concat(adatas_pred, join="outer")

    # reconstruct and reproject
    if latent_space == "pca":
        # reconstruct PCA
        cfp.pp.reconstruct_pca(adata_pred, use_rep="X_pca", ref_adata=adata_train)
        adata_pred.X = csr_matrix(adata_pred.layers["X_recon"])
        del adata_pred.layers["X_recon"]
    else:
        # reconstruct VAE
        adata_pred.layers["counts"] = csr_matrix(np.zeros(adata_pred.X.shape))
        vae_train.setup_anndata(adata_pred, layer="counts")
        X_recon = vae_train.get_reconstructed_expression(adata_pred, use_rep=sample_rep)
        X_recon = X_recon * adata_train.layers["counts"].sum(axis=1).mean()
        adata_pred.layers["X_recon_mean"] = X_recon
        adata_pred.layers["counts"] = csr_matrix(np.rint(X_recon).astype(int))
        adata_pred.X = adata_pred.layers["counts"]
        sc.pp.normalize_total(adata_pred, target_sum=1e4)
        sc.pp.log1p(adata_pred)
        # reproject VAE
        vae_eval.setup_anndata(adata_pred, layer="counts")
        adata_pred.obsm["X_vae_reproj"] = vae_eval.get_latent_representation(adata_pred)
    # reproject PCA
    cfp.pp.project_pca(adata_pred, adata, obsm_key_added="X_pca_reproj")

    # compute cfp metrics (e-distance, mmd, r2, sinkhorn)
    if minimal:
        metrics_cfp = compute_cfp_metrics(
            adata_eval, adata_pred, "X_pca_all", "X_pca_reproj", pred_name="pca_reproj"
        )
    else:
        metrics_dfs = [
            compute_cfp_metrics(
                adata_eval, adata_pred, sample_rep, sample_rep, fast=False
            ),
            compute_cfp_metrics(
                adata_eval, adata_pred, "X", "X", pred_name="recon", fast=False
            ),
            compute_cfp_metrics(
                adata_eval,
                adata_pred,
                "X_pca_all",
                "X_pca_reproj",
                pred_name="pca_reproj",
                fast=False,
            ),
        ]
        if latent_space != "pca":
            metrics_dfs.append(
                compute_cfp_metrics(
                    adata_eval,
                    adata_pred,
                    vae_eval_rep,
                    "X_vae_reproj",
                    pred_name="vae_reproj",
                    fast=False,
                )
            )
        metrics_cfp = reduce(
            lambda left, right: pd.merge(left, right, on="condition"), metrics_dfs
        )
    if len(sweep_conds) > 0:
        metrics_cfp["sweep"] = metrics_cfp["condition"].isin(sweep_conds)

    # compute cluster metrics
    if not minimal:
        rep_gt = "X_pca" if latent_space == "pca" else sample_rep
        rep_pred = "X_pca_reproj" if latent_space == "pca" else sample_rep
        metrics_cluster = compute_cluster_metrics(
            adata,
            adata_pred,
            rep_gt,
            rep_pred,
            n_neighbors=[1, 10, 30],
            label_keys=["leiden_4", "final_clustering"],
        )

        # project predicted data on UMAP
        umap_fit_transform(adata, adata_pred, "X_pca", "X_pca_reproj")
        if latent_space != "pca":
            umap_fit_transform(
                adata, adata_pred, sample_rep, sample_rep, key_added="X_umap_vae"
            )

        # save outputs
        metrics_cfp.to_csv(f"{out_dir}/{out_prefix}_metrics_cfp.csv", index=False)
        metrics_cluster.to_csv(
            f"{out_dir}/{out_prefix}_metrics_cluster.csv", index=False
        )
        if save_model:
            cf.save(out_dir, out_prefix, overwrite=True)
        if save_adata:
            adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")

        # visualize predictions on a UMAP
        figs_dir = os.path.join(out_dir, "figs")
        plot_predictions_umap(
            adata, adata_pred, "X_umap", plot_umap_combs, plot_umap_conds, figs_dir
        )
        if latent_space != "pca":
            plot_predictions_umap(
                adata,
                adata_pred,
                "X_umap_vae",
                plot_umap_combs,
                plot_umap_conds,
                figs_dir,
                space_name="vae",
            )

        # visualize predictions on a heatmap
        plot_predictions_heatmap(
            adata,
            adata_train,
            adata_pred,
            plot_combs=plot_heatmap_combs,
            plot_conds=plot_heatmap_conds,
            out_dir=figs_dir,
            label_keys=["leiden_4", "final_clustering"],
        )

        # visualize condition embedding
        plot_condition_embedding(cf, adata, figs_dir)

    else:
        # save outputs
        metrics_cfp.to_csv(f"{out_dir}/{out_prefix}_metrics_cfp.csv", index=False)
        if save_model:
            cf.save(out_dir, out_prefix, overwrite=True)
        if save_adata:
            adata_pred.write_h5ad(f"{out_dir}/{out_prefix}_pred.h5ad")


if __name__ == "__main__":
    # load config and run OT-FM
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs(os.path.join(config["out_dir"], "figs"), exist_ok=True)
    run_otfm_ineurons(**config)
