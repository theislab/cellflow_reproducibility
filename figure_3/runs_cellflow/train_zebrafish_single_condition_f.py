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
from cfp.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
from cfp.metrics import compute_r_squared, compute_e_distance, compute_metrics_fast
from cfp.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca



def compute_metrics(adata_ref: ad.AnnData, adata_ref_for_ct_error: ad.AnnData, adata_pred: ad.AnnData, adata_ood_true: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_broad", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    ct_true = adata_ood_true.obs[cell_type_col].value_counts().to_frame()
    ct_true = ct_true / ct_true.sum()
    
    compute_wknn(ref_adata=adata_ref_for_ct_error, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref_for_ct_error, label_key=cell_type_col)
    ct_transferred_pred = adata_pred.obs[f"{cell_type_col}_transfer"].value_counts().to_frame()
    ct_transferred_pred/=ct_transferred_pred.sum()
    all_cell_types = list(set(ct_true.index).union(set(ct_transferred_pred.index)))
    df_all_cell_types = pd.DataFrame(index=all_cell_types, data=np.zeros((len(all_cell_types), 2)), columns=["true", "pred"])
    df_all_cell_types["true"] = ct_true
    df_all_cell_types["pred"] = ct_transferred_pred
    df_all_cell_types = df_all_cell_types.fillna(0.0)
    cell_type_fraction_error = np.abs(df_all_cell_types["true"] - df_all_cell_types["pred"]).sum()
    
    all_cell_types_true = list(adata_ood_true.obs[cell_type_col].value_counts()[adata_ood_true.obs[cell_type_col].value_counts()>min_cells_for_dist_metrics].index)
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_aligned", query_rep_key="X_aligned")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    
    e_distance = {}
    r_sq = {}
    mmd = {}
    sdiv_10 = {}
    sdiv_100 = {}
    n_cell_types_covered = 0
    for cell_type in all_cell_types_true: 
        dist_true = adata_ood_true[adata_ood_true.obs[f"{cell_type_col}"]==cell_type].obsm["X_aligned"]
        dist_pred = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type].obsm["X_aligned"]
        if len(dist_pred) == 0:
            continue
        n_cell_types_covered+=1
        r_sq[f"r_squared_{cell_type}"] = compute_r_squared(dist_true, dist_pred)
        e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
        mmd[f"mmd_{cell_type}"] = compute_scalar_mmd(dist_true, dist_pred)
        sdiv_10[f"div_10_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=10.0)
        sdiv_100[f"div_100_{cell_type}"] = compute_sinkhorn_div(dist_true, dist_pred, epsilon=100.0)

    fraction_cell_types_covered = n_cell_types_covered/len(all_cell_types_true)


    # standard metrics
    ood_r_squared = compute_r_squared(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"])
    ood_sdiv_10 = compute_sinkhorn_div(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"], epsilon=10.0)
    ood_sdiv_100 = compute_sinkhorn_div(adata_ood_true.obsm["X_aligned"], adata_pred.obsm["X_aligned"], epsilon=100.0)
    
    # metrics to return
    dict_to_log["fraction_cell_types_covered"] = fraction_cell_types_covered
    dict_to_log["cell_type_fraction_error"] = cell_type_fraction_error
    dict_to_log["mean_r_sq_per_cell_type"] = np.mean(list(r_sq.values()))
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(list(e_distance.values()))
    dict_to_log["mean_mmd_per_cell_type"] = np.mean(list(mmd.values()))
    dict_to_log["mean_sdiv_10_per_cell_type"] = np.mean(list(sdiv_10.values()))
    dict_to_log["mean_sdiv_100_per_cell_type"] = np.mean(list(sdiv_100.values()))
    dict_to_log["median_r_sq_per_cell_type"] = np.median(list(r_sq.values()))
    dict_to_log["median_e_distance_per_cell_type"] = np.median(list(e_distance.values()))
    dict_to_log["median_mmd_per_cell_type"] = np.median(list(mmd.values()))
    dict_to_log["median_sdiv_10_per_cell_type"] = np.median(list(sdiv_10.values()))
    dict_to_log["median_sdiv_100_per_cell_type"] = np.median(list(sdiv_100.values()))
    dict_to_log.update(r_sq)
    dict_to_log.update(e_distance)
    dict_to_log.update(mmd)
    dict_to_log.update(sdiv_10)
    dict_to_log.update(sdiv_100)
    dict_to_log["ood_r_squared"] = ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    dict_to_log["ood_sdiv_10"] = ood_sdiv_10
    dict_to_log["ood_sdiv_100"] = ood_sdiv_100
    return dict_to_log


@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    ood_cond = config_dict["dataset"]["ood_condition"]
    control_key = config["dataset"]["control_key"]

    adata_train = adata[adata.obs["condition"]!=ood_cond]
    adata_ood_true = adata[adata.obs["condition"]==ood_cond]
    adata_ctrl = adata[adata.obs[control_key].to_numpy()]
    adata_ctrl_subsetted = []
    for tp in adata_ctrl.obs["timepoint"].unique():
        adata_tmp = adata_ctrl[adata_ctrl.obs["timepoint"]==tp]
        sc.pp.subsample(adata_tmp, n_obs=30000)
        adata_ctrl_subsetted.append(adata_tmp)
    adata_ctrl = ad.concat(adata_ctrl_subsetted)

    adata_ood_true_subsetted = []
    for tp in adata_ood_true.obs["timepoint"].unique():
        adata_tmp = adata_ood_true[adata_ood_true.obs["timepoint"]==tp]
        if adata_tmp.n_obs > 30000:
            sc.pp.subsample(adata_tmp, n_obs=30000)
        adata_ood_true_subsetted.append(adata_tmp)
    adata_ood_true = ad.concat(adata_ood_true_subsetted)

    adata_train.uns = adata.uns.copy()
    adata_ctrl.uns = adata.uns.copy()
    adata_ood_true.uns = adata.uns.copy()



    cf = cfp.model.CellFlow(adata_train, solver="otfm")

    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    split_covariates = config_dict["dataset"]["split_covariates"]
    split_covariates = split_covariates if len(split_covariates) else None

    cf.prepare_data(
        sample_rep=config["dataset"]["sample_rep"],
        control_key=control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),
        sample_covariates=("logtime",),
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

    train_fraction_for_validation = config_dict["training"]["train_fraction_for_validation"]
    if train_fraction_for_validation > 0.0:
        adata_train_subsampled = sc.pp.subsample(adata_train, copy=True, fraction=train_fraction_for_validation)
        cf.prepare_validation_data(
            adata_train_subsampled,
            name="train",
            n_conditions_on_log_iteration=config_dict["training"]["test_n_conditions_on_log_iteration"],
            n_conditions_on_train_end=config_dict["training"]["test_n_conditions_on_log_iteration"],
        )


    n_cells_max_in_source_ood_at_log_iteration = config_dict["training"]["n_cells_max_in_source_ood_at_log_iteration"]
    if n_cells_max_in_source_ood_at_log_iteration > 0:
        adata_ctrl_at_log_iteration = sc.pp.subsample(adata_ctrl, copy=True, n_obs=n_cells_max_in_source_ood_at_log_iteration)
    else:
        adata_ctrl_at_log_iteration = adata_ctrl
    adata_ood = ad.concat((adata_ctrl_at_log_iteration, adata_ood_true))
    adata_ood.uns = adata.uns.copy()
   
    cf.prepare_validation_data(
        adata_ood,
        name="ood",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    wandb_callback = cfp.training.WandbLogger(project="cfp_otfm_zebrafish_single_condition_f", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    
    callbacks = [wandb_callback]
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )

    out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/single_condition_f"

    cf.save(out_dir, file_prefix=wandb.run.name)

    covariate_data = adata_ood_true.obs.drop_duplicates(subset=["condition"])
    preds = cf.predict(adata=adata_ctrl, sample_rep="X_aligned", condition_id_key="condition", covariate_data=covariate_data)
    
    adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
    
    for cond, array in preds.items():
        adata_ref_for_ct_error = adata[adata.obs["condition"]!=cond]
        adata_ood_true_condition = adata_ood_true[adata_ood_true.obs["condition"]==cond]
        
        
        obs_data = pd.DataFrame({
            'condition': [cond] * array.shape[0]
        })
        adata_ood_pred = ad.AnnData(X=np.empty((len(array),1)), obs=obs_data)
        adata_ood_pred.obsm["X_aligned"] = array
        adata_ood_pred.write_h5ad(os.path.join(out_dir, f"{wandb.run.name}_{cond}_pred.h5ad"))
        out = compute_metrics(adata_ref=adata, adata_ref_for_ct_error=adata_ref_for_ct_error, adata_pred=adata_ood_pred, adata_ood_true=adata_ood_true_condition)
        pd.DataFrame.from_dict(out, columns=[cond], orient="index").to_csv(os.path.join(out_dir, f"{wandb.run.name}_{cond}.csv"))
        wandb.log({cond: out})
    
    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
