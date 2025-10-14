import functools
import os
import sys
import traceback
from typing import Dict, Literal, Optional, Tuple

import cellflow
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
import h5py
import pandas as pd
import pickle


@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)
    with h5py.File(config_dict['dataset']['dataset_path'], "r") as f:
        adata_all = ad.AnnData(obs=ad.io.read_elem(f["obs"]),
                            var=ad.io.read_elem(f["var"]),
                            varm = ad.io.read_elem(f["varm"]),
                            uns = ad.io.read_elem(f["uns"]),
                            obsm = ad.io.read_elem(f["obsm"]),
                            )

    df_cl_emb = pd.read_csv("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/cell_line_embedding_full_ccle_300_scaled.csv")
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/id_to_cell_line.pkl", "rb") as f:
        id_to_cell_line = pickle.load(f)

    ccle_embs_tahoe = {}
    for cl in id_to_cell_line.keys():
        ccle_embs_tahoe[cl] = df_cl_emb[df_cl_emb["stripped_cell_line_name"]==id_to_cell_line[cl]][[str(el) for el in np.arange(300)]].values.squeeze()

    adata_all.uns["cell_line_embeddings"] = ccle_embs_tahoe
    adata_all = adata_all[adata_all.obs["cell_line"].isin(ccle_embs_tahoe.keys())]

    ccle_embs_sciplex = {}
    for cl in ["A549", "K562", "MCF7"]:
        ccle_embs_sciplex[cl] = df_cl_emb[df_cl_emb["stripped_cell_line_name"]==cl][[str(el) for el in np.arange(300)]].values.squeeze()
    
    adata_train = adata_all
    #adata_ctrl = adata_all[adata_all.obs['control']]
    #sc.pp.subsample(adata_ctrl, n_obs=30000)
    
    cf = cellflow.model.CellFlow(adata_train, solver="otfm")

    # Prepare the training data and perturbation conditions
    perturbation_covariates = {k: tuple(v) for k, v in config_dict["dataset"]["perturbation_covariates"].items()}
    sample_covariates = list(config_dict["dataset"]["sample_covariates"]) if list(config_dict["dataset"]["sample_covariates"])[0] is not None else None
    sample_covariate_reps = dict(config_dict["dataset"]["sample_covariate_reps"]) if sample_covariates is not None else None

    print("Preparing data...")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=dict(config_dict["dataset"]["perturbation_covariate_reps"]),
        sample_covariates=sample_covariates,
        sample_covariate_reps=sample_covariate_reps,
        split_covariates=list(config_dict["dataset"]["split_covariates"]),
        #parallelize=True,
        #n_workers=2
    )
    print("Finished preparing data")

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
    print("Preparing model...")
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

    wandb_callback = cellflow.training.WandbLogger(project="cellflow_sciplex_hparam_new_embs", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)


    print("Begin training")
    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=[wandb_callback],
        valid_freq=config_dict["training"]["valid_freq"],
    )


    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    obs = adata_all.obs.copy()
    obs["condition"] = obs.apply(lambda x: f"{x['drug']}_{x['dosage']}_{x['cell_line']}", axis=1)
    obs = obs.drop_duplicates(subset="condition")
    
    df_mean, _ = cf.get_condition_embedding(obs[obs["drug"]!= "DMSO_TF"], condition_id_key="condition", rep_dict=adata_all.uns)
    df_mean.to_csv(os.path.join(config_dict["training"]["out_dir"], f"condition_embeddings_tahoe_{wandb.run.name}.csv"))
        
    split = 5
    adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
    adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
    adata_train_sci = sc.read_h5ad(adata_train_path)
    adata_ood_sci = sc.read_h5ad(adata_ood_path)
    adata_train_obs = adata_train_sci.obs.drop_duplicates(subset="condition")
    adata_ood_obs = adata_ood_sci.obs.drop_duplicates(subset="condition")
    sciplex_obs = pd.concat((adata_train_obs, adata_ood_obs))
    sciplex_obs["dosage"] = sciplex_obs["logdose"]
    sciplex_rep_dict = adata_train.uns
    sciplex_rep_dict["cell_line_embeddings"] = ccle_embs_sciplex
    sciplex_rep_dict["drug_embeddings"] = {k: np.concatenate((v, np.array([0]))) for k,v in sciplex_rep_dict["ecfp_dict"].items()}
    df_mean_sci, df_logvar_sci = cf.get_condition_embedding(sciplex_obs[sciplex_obs["drug"]!= "Vehicle"], condition_id_key="condition", rep_dict=sciplex_rep_dict)
    df_mean_sci["condition"] = df_mean_sci.index
    df_mean_sci["cell_line"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[0], axis=1)
    df_mean_sci["dose"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[-1], axis=1)
    df_mean_sci["drug"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[1], axis=1)
    df_mean_sci.to_csv(os.path.join(config_dict["training"]["out_dir"], f"condition_embeddings_sciplex_{wandb.run.name}.csv"))

    new_rows = []
    constant_dose = 5.0

    for drug in obs["drug"].unique():
        for cell_line in obs["cell_line"].unique():
            new_rows.append([drug, cell_line, constant_dose])

    new_obs = pd.DataFrame(new_rows, columns=["drug", "cell_line", "dosage"])
    new_obs["condition"] = new_obs.apply(lambda x: f"{x['drug']}_{x['dosage']}_{x['cell_line']}", axis=1)
    new_obs["control"] = False
    df_mean2, _ = cf.get_condition_embedding(new_obs[new_obs["drug"]!= "DMSO_TF"], condition_id_key="condition", rep_dict=adata_all.uns)
    df_mean2["condition"] = df_mean2.index
    df_mean2["cell_line"] = df_mean2.apply(lambda x: x["condition"].split("_")[-1], axis=1)
    df_mean2["dose"] = df_mean2.apply(lambda x: x["condition"].split("_")[1], axis=1)
    df_mean2["drug"] = df_mean2.apply(lambda x: x["condition"].split("_")[0], axis=1)
    mean_drug_emb = df_mean2.groupby("drug")[np.arange(64)].mean()
    mean_drug_emb.to_csv(os.path.join(config_dict["training"]["out_dir"], f"mean_drug_embeddings_from_tahoe_{wandb.run.name}.csv"))
    
    closest_cell_line = {}
    for cell_line in ["K562", "MCF7", "A549"]:
        cell_line_emb = ccle_embs_sciplex[cell_line]
        closest_emb = None
        closest_dist = np.inf
        for cl, cl_emb in ccle_embs_tahoe.items():
            print(cl, np.mean((cell_line_emb - cl_emb)**2))
            if np.mean((cell_line_emb - cl_emb)**2):
                closest_dist = np.mean((cell_line_emb - cl_emb)**2)
                closest_emb = cl
        closest_cell_line[cell_line] = closest_emb

    for k,v in closest_cell_line.items():
        sciplex_rep_dict["cell_line_dict"][k] = ccle_embs_tahoe[v]

    df_mean_sci, df_logvar_sci = cf.get_condition_embedding(sciplex_obs[sciplex_obs["drug"]!= "Vehicle"], condition_id_key="condition", rep_dict=sciplex_rep_dict)
    df_mean_sci["condition"] = df_mean_sci.index
    df_mean_sci["cell_line"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[0], axis=1)
    df_mean_sci["dose"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[-1], axis=1)
    df_mean_sci["drug"] = df_mean_sci.apply(lambda x: x["condition"].split("_")[1], axis=1)
    df_mean_sci.to_csv(os.path.join(config_dict["training"]["out_dir"], f"condition_embeddings_sciplex_closest_cell_line_{wandb.run.name}.csv"))
    

    return 1.0

if __name__ == "__main__":
    print(1)
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        import IPython; IPython.embed()
        sys.exit(1)