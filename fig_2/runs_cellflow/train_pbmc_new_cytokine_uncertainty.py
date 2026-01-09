import numpy as np
import pandas as pd
import seaborn as sns
import jax
import functools
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import flax.linen as nn
import optax
import cellflow
from cellflow.model import CellFlow
import cellflow.preprocessing as cfpp
from cellflow.utils import match_linear
from cellflow.plotting import plot_condition_embedding
from cellflow.preprocessing import transfer_labels, compute_wknn, centered_pca, project_pca, reconstruct_pca
from cellflow.metrics import compute_r_squared, compute_e_distance


def compute_frechet_var(x: dict[int, np.ndarray]):
    pairwise_e_dists = []
    denominator = 2 * len(x)**2
    for s1 in x.values():
        for s2 in x.values():
            pairwise_e_dists.append(compute_e_distance(s1, s2)**2)
    return np.sum(np.array(pairwise_e_dists))/denominator

def compute_metrics(adata_ref: ad.AnnData, adata_pred_all_samples: ad.AnnData, donor_deg_dict: dict, adata_ood_true: ad.AnnData, adata_ctrl: ad.AnnData, n_neighbors: int=1, cell_type_col: str = "cell_type_new", min_cells_for_dist_metrics: int = 50) -> dict:
    dict_to_log = {}
    compute_wknn(ref_adata=adata_ref, query_adata=adata_pred, n_neighbors=n_neighbors, ref_rep_key="X_pca", query_rep_key="X_pca_for_ct_transfer")
    transfer_labels(query_adata=adata_pred, ref_adata=adata_ref, label_key=cell_type_col)
    
    ood_e_distances = []
    decoded_ood_r_squareds = []
    mean_decoded_r_sq_per_cell_types = []
    mean_e_distance_per_cell_types = []
    mean_deg_r_sq_per_cell_types = []

    for i in range(10):

        adata_pred = adata_pred_all_samples[adata_pred_all_samples.obs["sample"]==i]
        # standard metrics
        ood_e_distance = compute_e_distance(adata_ood_true.obsm["X_pca"], adata_pred.obsm["X_pca"])
        decoded_ood_r_squared = compute_r_squared(adata_ood_true.X.toarray(), adata_pred.layers["X_recon"])
        
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
            
            deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
            deg_true_decoded = adata_true_ct[:,deg_mask].X.toarray()
            deg_pred_decoded = adata_pred_ct[:,deg_mask].X
            deg_r_sq[f"deg_decoded_r_squared_{cell_type}"] = compute_r_squared(deg_true_decoded, deg_pred_decoded)

        ood_e_distances.append(ood_e_distance)
        decoded_ood_r_squareds.append(decoded_ood_r_squared)
        mean_decoded_r_sq_per_cell_types.append(np.mean(list(r_sq.values())))
        mean_e_distance_per_cell_types.append(np.mean(list(e_distance.values())))
        mean_deg_r_sq_per_cell_types.append(np.mean(list(deg_r_sq.values())))

    # metrics to return
    dict_to_log["ood_e_distance"] = np.mean(ood_e_distances)
    dict_to_log["decoded_ood_r_squared"] = np.mean(decoded_ood_r_squareds)

    dict_to_log["mean_decoded_r_sq_per_cell_type"] = np.mean(mean_decoded_r_sq_per_cell_types)
    dict_to_log["mean_e_distance_per_cell_type"] = np.mean(mean_e_distance_per_cell_types)
    dict_to_log["mean_deg_r_sq_per_cell_type"] = np.mean(mean_deg_r_sq_per_cell_types)

    distr_dict = {}
    distr_dict_per_cell_type = {}
    gex_dict = {}
    gex_dict_per_cell_type = {}
    deg_gex_per_cell_type = {}
    for sample_idx in range(10):
        adata_pred = adata_pred_all_samples[adata_pred_all_samples.obs["sample"]==sample_idx]
        distr_dict[sample_idx] = adata_pred.obsm["X_pca"]
        gex_dict[sample_idx] = np.mean(adata_pred.X, axis=0)
        for ct in adata_pred.obs[f"{cell_type_col}_transfer"].unique():
            adata_pred_ct = adata_pred[adata_pred.obs[f"{cell_type_col}_transfer"]==ct]
            if ct not in distr_dict_per_cell_type:
                distr_dict_per_cell_type[ct] = {}
            distr_dict_per_cell_type[ct][sample_idx] = adata_pred_ct.obsm["X_pca"]
            if ct not in gex_dict_per_cell_type:
                gex_dict_per_cell_type[ct] = {}
            gex_dict_per_cell_type[ct][sample_idx] = np.mean(adata_pred_ct.X, axis=0)
            if ct not in deg_gex_per_cell_type:
                deg_gex_per_cell_type[ct] = {}
            deg_mask = [True if el in donor_deg_dict[ct_cyto] else False for el in adata_ood_true.var_names]
            deg_gex_per_cell_type[ct][sample_idx] = np.mean(adata_pred_ct[:,deg_mask].X, axis=0)
            

    dict_to_log["frechet_variance"] = compute_frechet_var(distr_dict)
    dict_to_log["var_mean_gex"] = np.var(np.stack(list(gex_dict.values())), axis=0).mean()
    f_var_per_cell_types = []

    for ct, distr_ct in distr_dict_per_cell_type.items():
        f_var_ct = compute_frechet_var(distr_ct)
        f_var_per_cell_types.append(f_var_ct)
        dict_to_log[f"frechet_variance_{ct}"] = f_var_ct

    dict_to_log["mean_frechet_variance_per_cell_type"] = np.mean(f_var_per_cell_types)
    
    var_mean_gex_per_cell_types = []
    var_mean_deg_gex_per_cell_types = []
    for ct, gex_ct in gex_dict_per_cell_type.items():
        var_mean_gex_ct = np.var(np.stack(list(gex_ct.values())), axis=0).mean()
        var_mean_gex_per_cell_types.append(var_mean_gex_ct)
        dict_to_log[f"var_mean_gex_{ct}"] = var_mean_gex_ct
        deg_gex_ct = deg_gex_per_cell_type[ct]
        var_mean_deg_gex_ct = np.var(np.stack(list(deg_gex_ct.values())), axis=0).mean()
        var_mean_deg_gex_per_cell_types.append(var_mean_deg_gex_ct)
        dict_to_log[f"var_mean_deg_gex_{ct}"] = var_mean_deg_gex_ct

    dict_to_log["mean_var_mean_gex_per_cell_type"] = np.mean(var_mean_gex_per_cell_types)
    dict_to_log["mean_var_mean_deg_gex_per_cell_type"] = np.mean(var_mean_deg_gex_per_cell_types)

    
    return dict_to_log




@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    cytokine_held_out = config_dict["dataset"]["cytokine_held_out"]
    idx_given_cytokine = str(config_dict["dataset"]["idx_given_cytokine"])
    control_key = "is_control"

    adata_base = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_base_{cytokine_held_out}.h5ad")
    adata_rest = sc.read_h5ad(f"/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine/adata_rest_{cytokine_held_out}.h5ad")
    cfp.preprocessing.centered_pca(adata_base, n_comps=100, method="rapids", keep_centered_data=False)
    cfp.preprocessing.project_pca(query_adata=adata_rest, ref_adata=adata_base)
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
        if adata_tmp.n_obs > 10000:
            sc.pp.subsample(adata_tmp, n_obs=10000)
        adata_ctrl_subsetted.append(adata_tmp)
    adata_ctrl = ad.concat(adata_ctrl_subsetted)

    
    adata_train.uns = adata_train.uns.copy()
    adata_ctrl.uns = adata_train.uns.copy()
    adata_ood_perturbed.uns = adata_train.uns.copy()


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
        flow=flow,
        layer_norm_before_concatenation=config_dict["model"]["layer_norm_before_concatenation"],
        linear_projection_before_concatenation=config_dict["model"]["linear_projection_before_concatenation"],
    )

    adata_ood = ad.concat((adata_ctrl, adata_ood_perturbed))
    adata_ood.uns = adata_train.uns.copy()
    
    metrics_callback = cfp.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cfp.training.PCADecodedMetrics(ref_adata=adata_base, metrics=["r_squared"])
    wandb_callback = cfp.training.WandbLogger(project="pbmc_with_uncertainty", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)
    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]
    
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    covariate_data = adata_ood_perturbed.obs.drop_duplicates(subset=["condition"])

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

    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/degs.pkl", "rb") as pickle_file:
        deg_genes = pickle.load(pickle_file)

    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/idcs_to_keep.pkl", "rb") as pickle_file:
        idcs_to_keep = pickle.load(pickle_file)
    
    adata_full = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/pbmc_with_pca.h5ad")
    adata_ood_true = adata_full[(adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"]==cytokine)]
    

    for condition in adata_pred_all.obs["condition"].unique():
        cytokine=cytokine_held_out
        donor=condition.split("_")[0]
        adata_ood_true = adata_ood_true[(adata_ood_true.obs["donor"] == donor) & (adata_ood_true.obs["cytokine"]==cytokine)]
        donor_deg_dict = {k: v for k, v in deg_genes.items() if (k.startswith(donor) and k.endswith(f"_{cytokine}"))}
        adata_pred = adata_pred_all[adata_pred_all.obs["condition"]==condition]
        adata_pred.uns["donors_in_train"] = list(adata_to_append.obs["donor"].unique())
        cfp.preprocessing.reconstruct_pca(query_adata=adata_pred, ref_adata=adata_base, use_rep="X_pca", layers_key_added = "X_recon")
        adata_pred.X = adata_pred.layers["X_recon"]
        cond_orig = condition
        adata_pred.write_h5ad(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}_preds.h5ad"))
        out = compute_metrics(adata_ref=adata_ref, adata_pred=adata_pred, donor_deg_dict=donor_deg_dict, adata_ood_true=adata_ood_true, adata_ctrl=adata_ctrl)
    
        pd.DataFrame.from_dict(out, columns=[condition], orient="index").to_csv(os.path.join(config_dict["training"]["out_dir"], f"{wandb.run.name}_{condition}.csv"))
        wandb.log({condition: out})
    
    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
