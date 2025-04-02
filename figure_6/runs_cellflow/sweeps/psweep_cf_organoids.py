import sys
import warnings

warnings.simplefilter(action="ignore")

import os

os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import yaml
import cloudpickle
import numpy as np

import jax
import jax.tree as jt
import jax.numpy as jnp
import flax.linen as nn
import optax
import scanpy as sc
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from functools import partial

import cfp
from cfp.utils import match_linear
from cfp.metrics._metrics import compute_metrics

import wandb
import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch

#### Project name & dir ####
PROJECT_NAME = "psweep_organoids_common"
RESULTS_DIR = f"/home/fleckj/projects/cellflow/results/"

#### Define splits ####
SPLIT_DATA = {
    "transfer_fatima_BMP4": {
        "train": "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/transfer/fatima/BMP4/adata_train.h5ad",
        "test": "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/transfer/fatima/BMP4/adata_test.h5ad",
    },
}


#### Read config ####
config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
print(config)

#### Connect to Ray and WandB ####
head_node = str(os.environ["head_node"])
port = str(os.environ["port"])
print(f"Connecting to head node: {head_node}:{port}")
ray.init(address=head_node + ":" + port)


#### Setup sweep config ####
fixed_params = {
    "iterations": 1e6,
    "lr_init": 0.00002,
    "b1": 0.9,
    "b2": 0.999,
    "batch_size": 512,
    "grad_accumulation_steps": 50,
    "cond_net": "mlp",
    "cond_pooling": "mean",
    "cond_n_layers_before_pool": 2,
    "cond_n_layers_after_pool": 2,
    "cond_dropout_before_pool": 0.0,
    "cond_pool_sample_covariates": True,
    "vf_n_frequencies": 1024,
    "vf_n_layers_decoder": 3,
    "vf_dropout_t": 0.0,
    "vf_n_layers_t": 2,
    "vf_t_hidden_dim": 1024,
    "vf_t_embed_dim": 1024,
    "flow_noise_type": "constant_noise",
    "flow_noise": 0.0,
    "ot_match_epsilon": 1.0,
    "ot_match_tau_a": 1.0,
    "ot_match_tau_b": 1.0,
}

screen_params = {k: tune.choice(v) for k, v in config.items() if k not in fixed_params}
config = {**fixed_params, **screen_params}

config["project"] = PROJECT_NAME
config["results_path"] = RESULTS_DIR
config["split_data"] = SPLIT_DATA


optuna_opt = OptunaSearch(
    metric="mmd",
    mode="min",
)


#### Train and evaluate functions ####
def train_cellflow(config, adata, adata_test):
    #### Train cellflow on training data ####
    OUT_DIR = config["out_dir"]

    # Init CellFlow and prepare data
    cf = cfp.model.CellFlow(adata, solver="otfm")

    cond_keys = list(adata.uns["conditions"].keys())
    control_key = "CTRL"
    sample_rep = "X_latent"

    # Train data
    cf.prepare_data(
        sample_rep=sample_rep,
        control_key=control_key,
        perturbation_covariates={"conditions": cond_keys},
        perturbation_covariate_reps={"conditions": "conditions"},
        sample_covariates=["dataset"],
        sample_covariate_reps={"dataset": "dataset_onehot"},
        split_covariates=["dataset"],
    )

    # Val data
    cf.prepare_validation_data(
        adata_test,
        name="validation",
        n_conditions_on_log_iteration=None,  # all conditions
        n_conditions_on_train_end=None,  # all conditions
    )

    #### Prepare model ####s
    # Condition encoder params
    if config["cond_net"] == "mlp":
        cond_encoder_kwargs_before_pool = [
            {
                "layer_type": "mlp",
                "dims": [config["cond_hidden_dim_before_pool"]]
                * config["cond_n_layers_before_pool"],
                "dropout_rate": config["cond_dropout_before_pool"],
            },
        ]
    elif config["cond_net"] == "self_attention":
        cond_encoder_kwargs_before_pool = [
            {
                "layer_type": "self_attention",
                "num_heads": [config["cond_hidden_num_heads"]]
                * config["cond_n_layers_before_pool"],
                "qkv_dim": [config["cond_hidden_dim_before_pool"]]
                * config["cond_n_layers_before_pool"],
                "dropout_rate": config["cond_dropout_before_pool"],
                "transformer_block": config["cond_hidden_trans_block"],
            },
        ]

    cond_encoder_kwargs_after_pool = [
        {
            "layer_type": "mlp",
            "dims": [config["cond_hidden_dim_after_pool"]]
            * config["cond_n_layers_after_pool"],
            "dropout_rate": config["cond_dropout_after_pool"],
        },
    ]

    if config["cond_pooling"] == "attention_token":
        cond_pooling_kwargs = {
            "num_heads": config["cond_pooling_num_heads"],
            "qkv_dim": config["cond_pooling_hidden_dim"],
            "dropout_rate": config["cond_pooling_dropout"],
        }
    elif config["cond_pooling"] == "attention_seed":
        cond_pooling_kwargs = {
            "num_heads": config["cond_pooling_num_heads"],
            "v_dim": config["cond_pooling_hidden_dim"],
            "seed_dim": config["cond_pooling_seed_dim"],
            "transformer_block": config["cond_pooling_trans_block"],
            "dropout_rate": config["cond_pooling_dropout"],
        }
    else:
        cond_pooling_kwargs = {}

    cond_encoder_kwargs = {
        "pooling": config["cond_pooling"],
        "pooling_kwargs": cond_pooling_kwargs,
        "layers_before_pool": cond_encoder_kwargs_before_pool,
        "layers_after_pool": cond_encoder_kwargs_after_pool,
        "cond_output_dropout": config["cond_output_dropout"],
    }

    # Velocity field params
    time_encoder_dims = [config["vf_t_hidden_dim"]] * config["vf_n_layers_t"] + [
        config["vf_t_embed_dim"]
    ]
    x_encoder_dims = [config["vf_x_hidden_dim"]] * config["vf_n_layers_x"] + [
        config["vf_x_embed_dim"]
    ]
    decoder_dims = [config["vf_decoder_hidden_dim"]] * config["vf_n_layers_decoder"]
    vf_kwargs = {
        "time_encoder_dims": time_encoder_dims,
        "hidden_dims": x_encoder_dims,
        "decoder_dims": decoder_dims,
        "time_encoder_dropout": config["vf_dropout_t"],
        "hidden_dropout": config["vf_dropout_x"],
        "decoder_dropout": config["vf_dropout_decoder"],
        "condition_embedding_dim": config["cond_embed_dim"],
        "time_freqs": int(config["vf_n_frequencies"]),
    }

    match_fn = partial(
        match_linear,
        epsilon=config["ot_match_epsilon"],
        tau_a=config["ot_match_tau_a"],
        tau_b=config["ot_match_tau_b"],
    )

    # Optimizer
    lr_schedule = optax.constant_schedule(config["lr_init"])
    optimizer = optax.MultiSteps(
        opt=optax.adam(learning_rate=lr_schedule, b1=config["b1"], b2=config["b2"]),
        every_k_schedule=config["grad_accumulation_steps"],
    )

    # Prepare model
    cf.prepare_model(
        **vf_kwargs,
        **cond_encoder_kwargs,
        pool_sample_covariates=config["cond_pool_sample_covariates"],
        flow={config["flow_noise_type"]: config["flow_noise"]},
        match_fn=match_fn,
        optimizer=optimizer,
    )

    #### Train model ####
    metrics_callback = cfp.training.Metrics(
        metrics=["r_squared", "e_distance", "mmd"],
        metric_aggregations=["mean", "median"],
    )

    wandb_callback = cfp.training.WandbLogger(
        project=config["project"], out_dir=OUT_DIR, config=config
    )

    cf.train(
        num_iterations=int(config["iterations"]),
        batch_size=int(config["batch_size"]),
        valid_freq=int(config["iterations"]) // 10,
        callbacks=[metrics_callback, wandb_callback],
        monitor_metrics=["validation_e_distance_mean", "validation_mmd_mean"],
    )

    os.makedirs(f"{OUT_DIR}/model", exist_ok=True)
    cf.save(f"{OUT_DIR}/model", overwrite=True)

    wandb.finish()
    return cf


def evaluate_cellflow(config, cf, adata, adata_test):
    #### Evaluate cellflow model ####
    OUT_DIR = config["out_dir"]

    test_conds = np.setdiff1d(adata_test.obs["condition"].unique(), "CTRL")
    gt_data = {
        condition: jnp.array(
            adata_test.obsm["X_latent"][adata_test.obs["condition"] == condition, :]
        )
        for condition in test_conds
    }
    adata_ctrl = adata[np.array(adata.obs["CTRL"].values), :].copy()
    perturb_obs = (
        adata_test.obs[~adata_test.obs["CTRL"]].iloc[:, :222].drop_duplicates()
    )

    pred_data = cf.predict(
        adata_ctrl,
        covariate_data=perturb_obs,
        sample_rep="X_latent",
        condition_id_key="condition",
    )

    all_metrics = jt.map(compute_metrics, gt_data, pred_data)
    all_metrics_df = pd.DataFrame(all_metrics).T

    all_metrics_df.to_csv(f"{OUT_DIR}/metrics.tsv", sep="\t")

    del gt_data

    #### Eval pred space ####
    pred_reps_arr = np.concatenate(list(pred_data.values()), axis=0)
    adata_pred = sc.AnnData(X=np.zeros((pred_reps_arr.shape[0], 1)))
    adata_pred.obsm["X_latent"] = pred_reps_arr
    adata_pred.obs["condition"] = np.repeat(
        list(pred_data.keys()), [v.shape[0] for v in pred_data.values()]
    )

    adata_pred.obs = (
        perturb_obs.set_index("condition")
        .loc[adata_pred.obs["condition"]]
        .reset_index()
    )

    adata_pred.obs["status"] = "pred"

    adata_pred.write_h5ad(f"{OUT_DIR}/test_predictions.h5ad")

    UMAP_PATH = "/home/fleckj/projects/cellflow/results/organoid_annots/umap.pkl"
    umap = cloudpickle.load(open(UMAP_PATH, "rb"))
    adata_pred.obsm["X_umap"] = umap.transform(adata_pred.obsm["X_latent"])

    adata_pred.write_h5ad(f"{OUT_DIR}/test_predictions.h5ad")

    #### Plot ####
    UMAP_PATH = "/home/fleckj/projects/cellflow/results/organoid_annots/umap.pkl"
    adata_pred = sc.read_h5ad(f"{OUT_DIR}/test_predictions.h5ad")
    test_conds = np.setdiff1d(adata_test.obs["condition"].unique(), "CTRL")

    sc.pp.subsample(adata, n_obs=20000)

    umap = cloudpickle.load(open(UMAP_PATH, "rb"))
    adata.obsm["X_umap"] = umap.transform(adata.obsm["X_latent"])
    adata_test.obsm["X_umap"] = umap.transform(adata_test.obsm["X_latent"])
    adata.obs["status"] = "gt"
    adata_test.obs["status"] = "gt"
    adata_pred.obs["status"] = "pred"

    adata_pred_gt = ad.concat(
        [
            adata[np.array(~adata.obs["CTRL"])],
            adata_test[np.array(~adata_test.obs["CTRL"])],
            adata_pred,
        ],
        join="outer",
    )

    os.makedirs(f"{OUT_DIR}/plots/", exist_ok=True)
    for cond in test_conds.tolist():
        ax, fig = plt.subplots(1, 2, figsize=(10, 5))

        adata_pred_gt.obs["plot_this"] = (
            (adata_pred_gt.obs["condition"] == cond)
            & (adata_pred_gt.obs["status"] == "pred")
        ).astype(int) + 0.5
        first_cell = adata_pred_gt.obs.index[0]
        adata_pred_gt.obs.loc[first_cell, "plot_this"] = 0

        p = sc.pl.umap(
            adata_pred_gt,
            color="plot_this",
            title=cond,
            show=False,
            color_map="RdPu",
            ax=fig[0],
        )

        adata_pred_gt.obs["plot_this"] = (
            (adata_pred_gt.obs["condition"] == cond)
            & (adata_pred_gt.obs["status"] == "gt")
        ).astype(int) + 0.5
        adata_pred_gt.obs.loc[first_cell, "plot_this"] = 0

        p = sc.pl.umap(
            adata_pred_gt,
            color="plot_this",
            title=cond,
            show=False,
            color_map="RdPu",
            ax=fig[1],
        )

        plt.tight_layout()
        cond_str = cond.replace("/", "_").replace(" ", "_")
        plt.savefig(f"{OUT_DIR}/plots/umap_{cond_str}.png")
        plt.close()

    return all_metrics_df


#### Define sweep function ####
def cellflow(config):
    #### Fixed parameters ####
    TRIAL_NAME = train.get_context().get_trial_name()
    config["trial_name"] = TRIAL_NAME

    PROJECT_DIR = os.path.join(config["results_path"], config["project"], TRIAL_NAME)

    all_metrics = []
    for split_type, data_paths in config["split_data"].items():
        print(f"Running split: {split_type}")

        #### Load and format data ####
        adata = sc.read_h5ad(data_paths["train"])
        adata_test = sc.read_h5ad(data_paths["test"])

        SPLIT_OUT_DIR = os.path.join(PROJECT_DIR, split_type)
        os.makedirs(SPLIT_OUT_DIR, exist_ok=True)

        split_config = {**config, "split_type": split_type, "out_dir": SPLIT_OUT_DIR}

        cf = train_cellflow(split_config, adata, adata_test)
        metrics_df = evaluate_cellflow(split_config, cf, adata, adata_test)

        metrics_df["split_type"] = split_type
        all_metrics.append(metrics_df)

        del adata, adata_test

    all_metrics_df = pd.concat(all_metrics)
    all_metrics_df.to_csv(f"{PROJECT_DIR}/all_metrics.tsv", sep="\t")

    # Return dict with mean metrics
    metrics_out = (
        all_metrics_df[["mmd", "e_distance", "split_type"]]
        .groupby("split_type")
        .mean()
        .mean()
        .to_dict()
    )
    return metrics_out


#### Run tuning with Ray ####
RESULTS_DIR = os.path.join(config["results_path"], "ray_results")

os.makedirs(RESULTS_DIR, exist_ok=True)

tuner = tune.Tuner(
    cellflow,
    param_space=config,
    tune_config=tune.TuneConfig(
        search_alg=optuna_opt,
        num_samples=500,
        max_concurrent_trials=4,
        metric="mmd",
        mode="min",
    ),
    run_config=train.RunConfig(
        storage_path=RESULTS_DIR,
        name=config["project"],
    ),
)
results = tuner.fit()

# Save results
with open(f"{RESULTS_DIR}/results.pkl", "wb") as f:
    cloudpickle.dump(results, f)

print("Done. Shutting down ray cluster")
ray.shutdown()
