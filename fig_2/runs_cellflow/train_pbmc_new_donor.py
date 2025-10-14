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
from cellflow.training import ComputationCallback
from cellflow.preprocessing import transfer_labels, compute_wknn
from cellflow.training import ComputationCallback
from numpy.typing import ArrayLike
from cellflow.metrics import compute_r_squared, compute_e_distance
from cellflow.metrics import compute_r_squared, compute_e_distance, compute_scalar_mmd, compute_sinkhorn_div
from cellflow.preprocessing import project_pca, reconstruct_pca, centered_pca
import gc
import cellflow
import pickle


def compute_metrics(adata_ref: ad.AnnData, adata_pred: ad.AnnData, donor_deg_dict: dict, adata_ood_true: ad.AnnData, adata_ctrl: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_new", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_pca", query_rep_key="X_pca_for_ct_transfer")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    
    e_distance = {}
    r_sq = {}
    mmd = {}
    sdiv_10 = {}
    sdiv_100 = {}
    deg_e_distance = {}
    deg_r_sq = {}
    deg_mmd = {}
    deg_sdiv_10 = {}
    deg_sdiv_100 = {}
    for ct_cyto in donor_deg_dict.keys(): 
        cell_type = ct_cyto.split("_")[1]
        adata_true_ct = adata_ood_true[(adata_ood_true.obs[f"{cell_type_col}"]==cell_type)]
        adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==cell_type]
        if adata_pred_ct.n_obs == 0:
            continue
        dist_true_decoded = adata_true_ct.X.toarray()
        dist_pred_decoded = adata_pred_ct.X
        dist_true = adata_true_ct.obsm["X_pca"]
        dist_pred = adata_pred_ct.obsm["X_pca"]
        r_sq[f"decoded_r_squared_{cell_type}"] = compute_r_squared(dist_true_decoded, dist_pred_decoded)
        e_distance[f"e_distance_{cell_type}"] = compute_e_distance(dist_true, dist_pred)
        mmd[f"mmd_{cell_type}"] = compute_scalar_mmd(dist_true, dist_pred)
        sdiv_10[f"div_10_{cell_type}"] = np.nan
        sdiv_100[f"div_100_{cell_type}"] = np.nan

        deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
        deg_true_decoded = adata_true_ct[:,deg_mask].X.toarray()
        deg_pred_decoded = adata_pred_ct[:,deg_mask].X
        deg_r_sq[f"deg_decoded_r_squared_{cell_type}"] = compute_r_squared(deg_true_decoded, deg_pred_decoded)
        deg_e_distance[f"deg_e_distance_{cell_type}"] = compute_e_distance(deg_true_decoded, deg_pred_decoded)
        deg_mmd[f"deg_mmd_{cell_type}"] = compute_scalar_mmd(deg_true_decoded, deg_pred_decoded)
        deg_sdiv_10[f"deg_div_10_{cell_type}"] = np.nan #compute_sinkhorn_div(deg_true_decoded, deg_pred_decoded, epsilon=10.0)
        deg_sdiv_100[f"deg_div_100_{cell_type}"] = np.nan #compute_sinkhorn_div(deg_true_decoded, deg_pred_decoded, epsilon=100.0)

    adata_concat = ad.concat([adata_ctrl, adata_pred], join="inner", label="all")
    sc.tl.rank_genes_groups(
            adata_concat,
            groupby="cytokine",
            reference="PBS",
            rankby_abs=True,
            n_genes=50,
            use_raw=False,
            method="wilcoxon",
        )
    predicted_deg_genes = [el[0] for el in list(adata_concat.uns["rank_genes_groups"]["names"])]

    # standard metrics
    decoded_ood_r_squared = compute_r_squared(adata_ood_true.X.toarray(), adata_pred.X)
    ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_mmd = compute_scalar_mmd(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
    ood_sdiv_10 = np.nan # compute_sinkhorn_div(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"], epsilon=10.0)
    ood_sdiv_100 = np.nan #compute_sinkhorn_div(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"], epsilon=100.0)
    
    # metrics to return
    dict_to_log["mean_decoded_r_sq_per_cell_type"] = np.mean(list(r_sq.values()))
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(list(e_distance.values()))
    dict_to_log["mean_mmd_per_cell_type"] = np.mean(list(mmd.values()))
    dict_to_log["mean_sdiv_10_per_cell_type"] = np.nan #np.mean(list(sdiv_10.values()))
    dict_to_log["mean_sdiv_100_per_cell_type"] = np.nan #np.mean(list(sdiv_100.values()))
    dict_to_log["mean_deg_r_sq_per_cell_type"] = np.mean(list(deg_r_sq.values()))
    dict_to_log["mean_deg_e_distance_per_cell_type"] = np.mean(list(deg_e_distance.values()))
    dict_to_log["mean_deg_mmd_per_cell_type"] = np.mean(list(deg_mmd.values()))
    dict_to_log["mean_deg_sdiv_10_per_cell_type"] = np.nan #np.mean(list(deg_sdiv_10.values()))
    dict_to_log["mean_deg_sdiv_100_per_cell_type"] = np.nan #np.mean(list(deg_sdiv_100.values()))
    
    dict_to_log.update(r_sq)
    dict_to_log.update(e_distance)
    dict_to_log.update(mmd)
    dict_to_log.update(sdiv_10)
    dict_to_log.update(sdiv_100)
    dict_to_log.update(deg_r_sq)
    dict_to_log.update(deg_e_distance)
    dict_to_log.update(deg_mmd)
    dict_to_log.update(deg_sdiv_10)
    dict_to_log.update(deg_sdiv_100)
    dict_to_log["decoded_ood_r_squared"] = decoded_ood_r_squared
    dict_to_log["ood_e_distance"] = ood_e_distance
    dict_to_log["ood_mmd"] = ood_mmd
    dict_to_log["ood_sdiv_10"] = ood_sdiv_10
    dict_to_log["ood_sdiv_100"] = ood_sdiv_100
    dict_to_log["predicted_deg_genes"] = predicted_deg_genes
    return dict_to_log

def add_embeddings(adata):
    with open('/lustre/groups/ml01/workspace/ot_perturbation/data/embeddings/pbmc_cytokine_mashup.pkl', 'rb') as f:
        mashup_emb = pickle.load(f)
    adata.uns["mashup"] = mashup_emb



@hydra.main(config_path="conf", config_name="train")
def run(config):
    print("starting")
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


    add_embeddings(adata_train)
    add_embeddings(adata_ctrl)
    add_embeddings(adata_ood_perturbed)


    print("now init cellflow")
    cf = cellflow.model.CellFlow(adata_train, solver="otfm")

    perturbation_covariates = {"cytokines": ["cytokine"]}
    split_covariates = ["donor"]
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key=control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps={"cytokines": config_dict["dataset"]["cytokine_embedding"]},
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
    probability_path = {config_dict["model"]["flow_type"]: config_dict["model"]["flow_noise"]}

    layers_before_pool = config_dict["model"]["layers_before_pool"]
    layers_after_pool = config_dict["model"]["layers_after_pool"]

    solver_kwargs = {"ema": config_dict["model"]["ema"]}
    time_max_period = config_dict["model"]["time_max_period"]
    time_max_period = time_max_period if time_max_period>0 else None

    # Prepare the model
    cf.prepare_model(
        condition_mode="deterministic",
        regularization=config_dict["model"]["regularization"],
        pooling=config_dict["model"]["pooling"],
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=config_dict["model"]["condition_embedding_dim"],
        cond_output_dropout=config_dict["model"]["cond_output_dropout"],
        time_freqs=config_dict["model"]["time_freqs"],
        time_max_period=time_max_period,
        time_encoder_dims=config_dict["model"]["time_encoder_dims"],
        time_encoder_dropout=config_dict["model"]["time_encoder_dropout"],
        hidden_dims=config_dict["model"]["hidden_dims"],
        hidden_dropout=config_dict["model"]["hidden_dropout"],
        conditioning=config_dict["model"]["conditioning"],
        decoder_dims=config_dict["model"]["decoder_dims"],
        decoder_dropout=config_dict["model"]["decoder_dropout"],
        probability_path=probability_path,
        solver_kwargs=solver_kwargs,
        match_fn=match_fn,
        optimizer=optimizer,
        layer_norm_before_concatenation=config_dict["model"]["layer_norm_before_concatenation"],
        linear_projection_before_concatenation=config_dict["model"]["linear_projection_before_concatenation"],
    )

    adata_ood = ad.concat((adata_ctrl, adata_ood_perturbed))
    adata_ood.uns = adata_train.uns.copy()

    cf.prepare_validation_data(
        adata_ood,
        name='ood',
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    
    metrics_callback = cellflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cellflow.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared"])
    wandb_callback = cellflow.training.WandbLogger(project="cellflow_new_donor_new", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]
    print('starting training')
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
            donor, cytokine = condition.split("_")
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
            adata_pred.X = adata_pred.layers["X_recon"].copy()
            adata_pred.var_names = adata_train.var_names
            adata_pred.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}_preds.h5ad"))
            cond_orig = condition
            condition = condition + "_" + str(len(adata_pred.uns["cytokine_in_train"]))
            
            out_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_new" #added _new
            adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
            adata_ood_true = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
            adata_ctrl = adata_full[(adata_full.obs["cytokine"]=="PBS") & (adata_full.obs["donor"]==donor)]
            with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
                idcs_to_keep = pickle.load(pickle_file)
            
            adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
            project_pca(query_adata=adata_pred, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")
            project_pca(query_adata=adata_pred, ref_adata=adata_full, obsm_key_added="X_pca")
            with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs.pkl", "rb") as pickle_file:
                deg_genes = pickle.load(pickle_file)
            donor_deg_dict = {k: v for k, v in deg_genes.items() if (k.startswith(donor) and k.endswith(f"_{cytokine}"))}
            adata_pred.obs["cytokine"] = cytokine
            adata_pred.obs["donor"] = donor
            
            out = compute_metrics(adata_ref=adata_ref, adata_pred=adata_pred, donor_deg_dict=donor_deg_dict, adata_ood_true=adata_ood_true, adata_ctrl=adata_ctrl)
            out["wandb_name"] = pred_file.split("_")[0]
            out["cytokine_in_train"] = adata_pred.uns["cytokine_in_train"]
            out["cytokine_in_train"] = len(adata_pred.uns["cytokine_in_train"])
            expr = f"{wandb.run.name}_{condition}"
            pd.DataFrame.from_dict(out, columns=[cond], orient="index").to_csv(os.path.join(out_dir, f"{expr}_metrics.csv"))
            
            
    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
