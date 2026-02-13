import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import flax.linen as nn
import pickle
import optax
import hydra
import wandb
import os
import cellflow
from cellflow.utils import match_linear
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance
from cellflow.data._dataloader import OOCTrainSampler, PredictionSampler, TrainSampler, ValidationSampler
import scipy 
from typing import Any
from cellflow.training import ComputationCallback
from cellflow._types import ArrayLike
from cellflow.solvers import _genot, _otfm
import jax.tree_util as jtu
import wandb
import numpy as np

def wandb_scatter(x, y, *, x_name: str, y_name: str, title: str, step: int | None = None):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    table = wandb.Table(data=[[float(xi), float(yi)] for xi, yi in zip(x, y)],
                        columns=[x_name, y_name])

    plot = wandb.plot.scatter(table, x_name, y_name, title=title)
    payload = {title: plot}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)
        
def compute_frechet_var(x: dict[int, np.ndarray]):
    pairwise_e_dists = []
    denominator = 2 * len(x)**2
    for s1 in x.values():
        for s2 in x.values():
            pairwise_e_dists.append(compute_e_distance(s1, s2)**2)
    return np.sum(np.array(pairwise_e_dists))/denominator
    
class VarCallback(ComputationCallback):
    def __init__(self, val_data: dict[str, ValidationSampler], ref_adata: ad.AnnData, n_draws: int = 10, predict_kwargs: dict[str, Any] = {}):
        self.n_draws = n_draws
        self.val_data = val_data
        self.pcs = ref_adata.varm["PCs"]
        self.means = ref_adata.varm["X_mean"]
        self.reconstruct_data = lambda x: x @ np.transpose(self.pcs) + np.transpose(self.means)
        self.predict_kwargs = predict_kwargs

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        return self.on_log_iteration(*args, **kwargs)

    def on_log_iteration(
        self, 
        valid_source_data: dict[str, dict[str, ArrayLike]],
        valid_true_data: dict[str, dict[str, ArrayLike]],
        valid_pred_data: dict[str, dict[str, ArrayLike]],
        solver: _otfm.OTFlowMatching | _genot.GENOT,
    ) -> dict[str, float]:
        valid_source_data: dict[str, dict[str, ArrayLike]] = {}
        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}

        
        metrics = {}
        for val_key, vdl in self.val_data.items():
            mean_e_dists = []
            mean_r_sqs = []
            frechet_var = []
            var_gex = []
            batch = vdl.sample(mode="on_log_iteration")
            for cond in batch["source"].keys():
                e_distances = []
                r_sqs = []
                src = batch["source"][cond]
                condition = batch["condition"][cond]
                true_tgt = batch["target"][cond]
                valid_pred_data[cond] = {}
                for i in range(self.n_draws):
                    valid_pred_data[cond][i] = solver.predict(src, condition=condition,rng=jax.random.PRNGKey(i), **self.predict_kwargs)
                frechet_var.append(compute_frechet_var(valid_pred_data[cond]))
                for i in range(self.n_draws):
                    e_distances.append(compute_e_distance(true_tgt, valid_pred_data[cond][i]))
                valid_true_data_decoded = self.reconstruct_data(true_tgt)
                valid_pred_data_decoded = jtu.tree_map(self.reconstruct_data, valid_pred_data[cond])
                var_gex.append(np.var(np.stack(list(valid_pred_data_decoded.values())), axis=0).mean())
                for i in range(self.n_draws):
                    r_sqs.append(compute_r_squared(valid_true_data_decoded, valid_pred_data_decoded[i]))
                
                mean_r_sqs.append(np.mean(r_sqs))
                mean_e_dists.append(np.mean(e_distances))
                print(mean_r_sqs, r_sqs)

            metrics[f"{val_key}_e_dist"] = np.mean(mean_e_dists)
            metrics[f"{val_key}_r_sq"] = np.mean(mean_r_sqs)
            metrics[f"{val_key}_pearson_calibration"] = np.corrcoef(np.array(mean_e_dists), np.array(frechet_var))[0,1]
            metrics[f"{val_key}_spearman_calibration"] = scipy.stats.spearmanr(np.array(mean_e_dists), np.array(frechet_var)).statistic
            metrics[f"{val_key}_pearson_decoded_calibration"] = np.corrcoef(np.array(mean_r_sqs), -np.array(var_gex))[0,1]
            metrics[f"{val_key}_spearman_decoded_calibration"] = scipy.stats.spearmanr(np.array(mean_r_sqs), -np.array(var_gex)).statistic

            wandb_scatter(
                x=frechet_var,
                y=mean_e_dists,
                x_name="frechet_var",
                y_name="mean_e_dist",
                title=f"{val_key}/calibration_e_dist_vs_frechet_var",
            )
            
            wandb_scatter(
                x=var_gex,
                y=mean_r_sqs,
                x_name="var_gex",
                y_name="mean_r_sq",
                title=f"{val_key}/decoded_calibration_r_sq_vs_var_gex",
            )

            
        return metrics


@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    cytokine_held_out = config_dict["dataset"]["cytokine_held_out"]
    idx_given_cytokine = str(config_dict["dataset"]["idx_given_cytokine"])
    control_key = "is_control"

    adata_base = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_base_{cytokine_held_out}.h5ad")
    adata_rest = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_rest_{cytokine_held_out}.h5ad")
    cellflow.preprocessing.centered_pca(adata_base, n_comps=100, method="rapids", keep_centered_data=False)
    cellflow.preprocessing.project_pca(query_adata=adata_rest, ref_adata=adata_base)
    donors_to_impute = adata_rest.uns["split_info"][idx_given_cytokine]["donors_to_impute"]
    donors_to_train_data = adata_rest.uns["split_info"][idx_given_cytokine]["donors_to_train_data"]
    adata_to_append = adata_rest[adata_rest.obs["donor"].isin(donors_to_train_data)]
    adata_train = ad.concat((adata_base, adata_to_append))
    adata_train.uns = adata_base.uns.copy()
    adata_ood_perturbed = adata_rest[adata_rest.obs["donor"].isin(donors_to_impute)]
    
    adata_ctrl = adata_train[adata_train.obs[control_key].to_numpy()]
    
    adata_ctrl_subsetted = []
    for donor in adata_ctrl.obs["donor"].unique():
        adata_tmp = adata_ctrl[adata_ctrl.obs["donor"]==donor]
        if adata_tmp.n_obs > 1000:
            sc.pp.subsample(adata_tmp, n_obs=1000)
        adata_ctrl_subsetted.append(adata_tmp)
    adata_ctrl = ad.concat(adata_ctrl_subsetted)

    adata_ood_subsetted = []
    for cond in adata_ood_perturbed.obs["condition"].unique():
        adata_tmp = adata_ood_perturbed[adata_ood_perturbed.obs["condition"]==cond]
        if adata_tmp.n_obs > 1000:
            sc.pp.subsample(adata_tmp, n_obs=1000)
        adata_ood_subsetted.append(adata_tmp)
    adata_ood_perturbed = ad.concat(adata_ood_subsetted)

    
    adata_train.uns = adata_train.uns.copy()
    adata_ctrl.uns = adata_train.uns.copy()
    


    cf = CellFlow(adata_train, solver="otfm")

    perturbation_covariates = {"cytokines": ["cytokine"]}
    split_covariates = ["donor"]
    
    cf.prepare_data(
        sample_rep="X_pca",
        control_key=control_key,
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps={"cytokines": "esm2_embeddings"},
        sample_covariates=("donor",),
        sample_covariate_reps={"donor": "donor_one_hot"},
        split_covariates=split_covariates,
    )

    adata_ood = ad.concat((adata_ctrl[adata_ctrl.obs["donor"].isin(donors_to_impute)], adata_ood_perturbed))
    adata_ood.uns = adata_train.uns.copy()

    cf.prepare_validation_data(
        adata_ood,
        name="ood",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
    )
    

    match_fn = functools.partial(
        match_linear,
        epsilon=config_dict["model"]["epsilon"],
        scale_cost="mean",
        tau_a=config_dict["model"]["tau_a"],
        tau_b=config_dict["model"]["tau_b"]
    )
    optimizer = optax.MultiSteps(optax.adam(config_dict["model"]["learning_rate"]), config_dict["model"]["multi_steps"])
    probability_path = {config_dict["model"]["flow_type"]: config_dict["model"]["flow_noise"]}

    layers_before_pool = config_dict["model"]["layers_before_pool"]
    layers_after_pool = config_dict["model"]["layers_after_pool"]


    # Prepare the model
    cf.prepare_model(
        condition_mode="stochastic",
        regularization=config_dict["model"]["regularization"],
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
        probability_path=probability_path,
        layer_norm_before_concatenation=config_dict["model"]["layer_norm_before_concatenation"],
        linear_projection_before_concatenation=config_dict["model"]["linear_projection_before_concatenation"],
    )

    cb = VarCallback({"ood": ValidationSampler(cf.validation_data["ood"])}, ref_adata = adata_base, n_draws=10)

    wandb_callback = cellflow.training.WandbLogger(project="pbmc_with_uncertainty_2", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    callbacks = [cb, wandb_callback]

    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    
    
    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    covariate_data = adata_ood_perturbed.obs.drop_duplicates(subset=["condition"])

    preds = []
    for i in range(10):
        preds.append(cf.predict(adata=adata_ctrl, sample_rep="X_pca", condition_id_key="condition", rng=jax.random.PRNGKey(i), covariate_data=covariate_data))

    
    
    adatas_preds_all = []
    for i, pred in enumerate(preds):
        adata_preds = []
        for cond, array in pred.items():
        
            obs_data = pd.DataFrame({
                'condition': [cond] * array.shape[0]
            })
            adata_pred = ad.AnnData(X=np.empty((len(array),adata_train.n_vars)), obs=obs_data)
            adata_pred.obsm["X_pca"] = np.squeeze(array)
            adata_preds.append(adata_pred)
        
        adata_preds = ad.concat(adata_preds)
        adata_preds.obs["sample"] = i
        adata_preds.var_names = adata_train.var_names
        adatas_preds_all.append(adata_preds)
    
    
    adata_pred_all = ad.concat(adatas_preds_all)
    reg = str(config_dict["model"]["regularization"])
    cellflow.preprocessing.reconstruct_pca(query_adata=adata_pred_all, ref_adata=adata_base, use_rep="X_pca", layers_key_added = "X_recon")
    adata_pred_all.X = adata_pred_all.layers["X_recon"]
    adata_pred_all.obsm["X_pca_train"] = adata_pred_all.obsm["X_pca"].copy()
    
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)

    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
   
    project_pca(query_adata=adata_pred_all, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")
    project_pca(query_adata=adata_pred_all, ref_adata=adata_full, obsm_key_added="X_pca")

    adata_pred_all.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{reg}_all_preds.h5ad"))
    

    """with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs.pkl", "rb") as pickle_file:
        deg_genes = pickle.load(pickle_file)

    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)

    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)]
    
    cellflow.preprocessing.reconstruct_pca(query_adata=adata_pred_all, ref_adata=adata_base, use_rep="X_pca", layers_key_added = "X_recon")
    adata_pred_all.X = adata_pred_all.layers["X_recon"]
    adata_pred_all.obsm["X_pca_train"] = adata_pred_all.obsm["X_pca"].copy()
    project_pca(query_adata=adata_pred_all, ref_adata=adata_ref, obsm_key_added="X_pca_for_ct_transfer")
    project_pca(query_adata=adata_pred_all, ref_adata=adata_full, obsm_key_added="X_pca")


    adata_pred_all.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_all_preds.h5ad"))
    adata_pred.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}_all_preds.h5ad"))
        
        
    #out_dicts = {}
    #for condition in adata_pred_all.obs["condition"].unique():
    #    cytokine=cytokine_held_out
    #    donor=condition.split("_")[0]
    #    adata_ood_true_red = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
    #    donor_deg_dict = {k: v for k, v in deg_genes.items() if (k.startswith(donor) and k.endswith(f"_{cytokine}"))}
    #    adata_pred = adata_pred_all[adata_pred_all.obs["condition"]==condition]
    #    adata_pred.uns["donors_in_train"] = list(adata_to_append.obs["donor"].unique())
    #    cond_orig = condition
    #    adata_pred.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}_preds.h5ad"))
    #    out = compute_metrics(adata_ref=adata_ref, adata_pred_all_samples=adata_pred_all, donor_deg_dict=donor_deg_dict, adata_ood_true=adata_ood_true_red, adata_ctrl=adata_ctrl)
    #    out_dicts[condition] = out
    #    pd.DataFrame.from_dict(out, columns=[condition], orient="index").to_csv(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}.csv"))
        
        
    #df = pd.DataFrame(columns=["e_distance", "e_distance_var", "r_sq_per_cell_type", "r_sq_per_cell_type_var", "deg_r_sq", "deg_r_sq_var"])
    #cal_log_dict = {}
    #for i, (k, v) in enumerate(out_dicts.items()):
    #    df.loc[i, :] = {
    #        "e_distance": v["ood_e_distance"],
    #        "e_distance_var": v["frechet_variance"],
    #        "r_sq_per_cell_type": v["mean_decoded_r_sq_per_cell_type"],
    #        "r_sq_per_cell_type_var": v["mean_var_mean_gex_per_cell_type"],
    #        "deg_r_sq": v["mean_deg_r_sq_per_cell_type"],
    #        "deg_r_sq_var": v["mean_var_mean_deg_gex_per_cell_type"],
    #    }
    #df["neg_deg_r_sq"] = 1-df["deg_r_sq"]
    #df["neg_r_sq_per_cell_type"] = 1-df["r_sq_per_cell_type_var"]
    #cal_log_dict = {}
    #cal_log_dict["e_dist_calibration"] = df[["e_distance", "e_distance_var"]].corr(method="spearman").iloc[0,1]
    #cal_log_dict["gex_calibration"] = df[["r_sq_per_cell_type", "r_sq_per_cell_type_var"]].corr(method="spearman").iloc[0,1]
    #cal_log_dict["deg_calibration"] = df[["neg_deg_r_sq", "deg_r_sq_var"]].corr(method="spearman").iloc[0,1]
    #cal_log_dict["mean_e_distance"] = df["e_distance"].mean()
    #cal_log_dict["mean_r_sq_per_cell_type"] = df["r_sq_per_cell_type"].mean()
    #cal_log_dict["mean_deg_r_sq"] = df["deg_r_sq"].mean()

    

    #wandb.log({condition: cal_log_dict})
    """
    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
