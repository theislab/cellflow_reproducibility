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
import pandas as pd
import pickle
import cellflow




@hydra.main(config_path="conf", config_name="train")
def run(config):
    config_dict  = OmegaConf.to_container(config, resolve=True)

    embedding_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/cellflow/embeddings_from_tahoe"

    def get_embeddings(type: str):
        if type == "closest_cell_line":
            with open(os.path.join(embedding_dir, "sci_cl_cell_line.pkl"), "rb") as f:
                embedding_dict = pickle.load(f)
                
        elif type == "direct_embedding":
            with open(os.path.join(embedding_dir, "sci_emb.pkl"), "rb") as f:
                embedding_dict = pickle.load(f)
        elif type == "concatenated_drug_embeddings":
            with open(os.path.join(embedding_dir, "sci_drug_concatenated_from_tahoe.pkl"), "rb") as f:
                embedding_dict = pickle.load(f)
        else:
            raise ValueError
        return embedding_dict

    split = config_dict["dataset"]["split"]
    adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
    adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
    adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
    adata_train = sc.read_h5ad(adata_train_path)
    adata_test = sc.read_h5ad(adata_test_path)
    adata_ood = sc.read_h5ad(adata_ood_path)

    adata_train.obs["cell_line_drug"] = adata_train.obs["cell_line"].astype("str") + "_" + adata_train.obs["drug"].astype("str")
    adata_ood.obs["cell_line_drug"] = adata_ood.obs["cell_line"].astype("str") + "_" + adata_ood.obs["drug"].astype("str")

    emb_type = config_dict["model"]["embedding_type"]
    drug_embeddings = get_embeddings(emb_type)

    if emb_type == "direct_embedding":
        adata_train.uns["cell_line_drug_embedding"] = drug_embeddings
        adata_ood.uns["cell_line_drug_embedding"] = drug_embeddings
        perturbation_covariates = {"cell_line_drug": ["cell_line_drug"], "dose": ["dose"]}
        perturbation_covariate_reps = {"cell_line_drug": "cell_line_drug_embedding"}
        sample_covariates = []
        sample_covariate_reps = {}
        split_covariates = ["cell_line"]
        layers_before_pool = {"cell_line_drug": 
                                {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                            "dose":
                                {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                            }

    elif emb_type == "closest_cell_line":
        adata_train.uns["cell_line_drug_embedding"] = drug_embeddings
        adata_ood.uns["cell_line_drug_embedding"] = drug_embeddings
        perturbation_covariates = {"cell_line_drug": ["cell_line_drug"], "dose": ["dose"]}
        perturbation_covariate_reps = {"cell_line_drug": "cell_line_drug_embedding"}
        sample_covariate_reps = {}
        sample_covariates = []
        split_covariates = ["cell_line"]
        layers_before_pool = {"cell_line_drug": 
                                {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                            "dose":
                                {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                            }

    elif emb_type == "concatenated_drug_embeddings":
        adata_train.uns["drug_embedding"] = drug_embeddings
        adata_ood.uns["drug_embedding"] = drug_embeddings
        
        df_cl_emb = pd.read_csv("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/cell_line_embedding_full_ccle_300_scaled.csv")
        with open("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/id_to_cell_line.pkl", "rb") as f:
            id_to_cell_line = pickle.load(f)
        ccle_embs_sciplex = {}
        for cl in ["A549", "K562", "MCF7"]:
            ccle_embs_sciplex[cl] = df_cl_emb[df_cl_emb["stripped_cell_line_name"]==cl][[str(el) for el in np.arange(300)]].values.squeeze()
        adata_train.uns["cell_line_embedding"] = ccle_embs_sciplex
        adata_ood.uns["cell_line_embedding"] = ccle_embs_sciplex
        perturbation_covariates = {"drug": ["drug"], "dose": ["dose"]}
        perturbation_covariate_reps = {"drug": "drug_embedding"}
        sample_covariates = ["cell_line"]
        sample_covariate_reps = {"cell_line": "cell_line_embedding"}
        split_covariates = ["cell_line"]
        layers_before_pool = {"drug": 
                                {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                            "dose":
                                {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                            "cell_line":
                                {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                            }


    else:
        raise ValueError

    
    adata_ood_1 = adata_ood[adata_ood.obs["dose"].isin((0.0, 10.0))]
    adata_ood_2 = adata_ood[adata_ood.obs["dose"].isin((0.0, 100.0))]
    adata_ood_3 = adata_ood[adata_ood.obs["dose"].isin((0.0, 1000.0))]
    adata_ood_4 = adata_ood[adata_ood.obs["dose"].isin((0.0, 10000.0))]
    

    cf = cellflow.model.CellFlow(adata_train, solver="otfm")

    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates=perturbation_covariates,
        perturbation_covariate_reps=perturbation_covariate_reps,
        sample_covariates=sample_covariates,
        sample_covariate_reps=sample_covariate_reps,
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

    cf.prepare_validation_data(
        adata_ood_1,
        name="ood_1",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_2,
        name="ood_2",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_3,
        name="ood_3",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )

    cf.prepare_validation_data(
        adata_ood_4,
        name="ood_4",
        n_conditions_on_log_iteration=config_dict["training"]["ood_n_conditions_on_log_iteration"],
        n_conditions_on_train_end=config_dict["training"]["ood_n_conditions_on_log_iteration"],
    )


    metrics_callback = cellflow.training.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_metrics_callback = cellflow.training.PCADecodedMetrics(ref_adata=adata_train, metrics=["r_squared", "mmd", "e_distance"])
    wandb_callback = cellflow.training.WandbLogger(project="cellflow_sciplex_with_tahoe_embeddings", out_dir="/home/icb/dominik.klein/tmp", config=config_dict)

    callbacks = [metrics_callback, decoded_metrics_callback, wandb_callback]

    cf.train(
        num_iterations=config_dict["training"]["num_iterations"],
        batch_size=config_dict["training"]["batch_size"],
        callbacks=callbacks,
        valid_freq=config_dict["training"]["valid_freq"],
    )
    if config_dict["training"]["save_model"]:
        cf.save(config_dict["training"]["out_dir"], file_prefix=wandb.run.name)

    
    return 1.0

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
