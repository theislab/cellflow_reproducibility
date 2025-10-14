import warnings
import os

warnings.simplefilter(action="ignore")
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
import re
from datetime import datetime
import subprocess as sp
import numpy as np
import pandas as pd
import yaml
import cloudpickle
import wandb
import ray
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.optuna import OptunaSearch


def get_ray_param(param):
    if isinstance(param, list):
        return tune.choice(param)
    else:
        return param


def run_otfm_bsub(config):
    out_dir = config["out_dir"]
    logs_dir = os.path.join(out_dir, "logs")
    configs_dir = os.path.join(out_dir, "configs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    # parse train-test splits from config
    split_keys = [
        "train_combs",
        "exclude_combs",
        "eval_combs",
        "sweep_combs",
        "train_conds",
        "eval_conds",
        "sweep_conds",
        "plot_umap_combs",
        "plot_umap_conds",
        "plot_heatmap_combs",
        "plot_heatmap_conds",
    ]
    n_splits = config.pop("n_splits", 1)
    split_params = {}
    for key, val in config.items():
        if key in split_keys:
            if val is not None:
                split_params[key] = val
            else:
                split_params[key] = []
    if len(split_params["train_combs"]) > 0 and len(split_params["exclude_combs"]) > 0:
        raise ValueError("Only one of `train_combs` and `exclude_combs` can be set.")

    # unify split parameters format
    for split_param, split_param_list in split_params.items():
        if split_param_list and all(isinstance(x, list) for x in split_param_list):
            if len(split_param_list) == 1:
                split_params[split_param] = split_param_list * n_splits
            elif len(split_param_list) != n_splits:
                raise ValueError(f"Length of {split_param} must be equal to 1.")
        else:
            split_params[split_param] = [split_param_list for _ in range(n_splits)]
    config.update({key: val[0] for key, val in split_params.items()})

    bsub_params = {
        key.split("_")[1]: config.pop(key)
        for key in list(config.keys())
        if key.startswith("bsub")
    }

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(np.random.randint(1000))
    )
    script = config.pop("script")
    run_config = os.path.join(configs_dir, run_id + ".yml")
    stout = os.path.join(logs_dir, run_id + ".out")
    sterr = os.path.join(logs_dir, run_id + ".err")
    config["out_prefix"] = run_id

    with open(run_config, "w") as f:
        f.write(yaml.dump(config, sort_keys=False))

    cmd = ["bsub"]
    cmd.extend([f'-{key} "{str(value)}"' for key, value in bsub_params.items()])
    cmd.extend(["-o", stout])
    cmd.extend(["-e", sterr])
    cmd.extend(["python", script])
    cmd.extend([run_config])
    cmd = " ".join(cmd)
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, _ = process.communicate()

    # Extract job ID from BSUB output
    stdout = stdout.decode("utf-8")
    job_id = int(stdout.split()[1].strip("<>"))
    with open(stout, "a") as f:
        f.write(f"Job ID: {job_id}\n")

    # Poll for job completion
    completed = False
    while not completed:
        status_cmd = f"bjobs {job_id}"
        status_process = sp.Popen(
            status_cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE
        )
        status_output, _ = status_process.communicate()
        status_output = status_output.split(b"\n")[1]
        if not (b"PEND" in status_output or b"RUN" in status_output):
            completed = True
        if not completed:
            time.sleep(60)

    metrics = pd.read_csv(f"{out_dir}/{run_id}_metrics_cfp.csv")
    metrics = metrics.drop(columns="condition")
    if "sweep" in metrics.columns:
        metrics = metrics[metrics["sweep"]]
    metrics_mean = metrics.mean(axis=0).to_dict()
    metrics_median = metrics.median(axis=0)
    metrics_median.index = [f"{i}_median" for i in metrics_median.index]
    metrics_std = metrics.std(axis=0)
    metrics_std.index = [f"{i}_std" for i in metrics_std.index]
    metrics_mean.update(metrics_median.to_dict())
    metrics_mean.update(metrics_std.to_dict())
    return metrics_mean


if __name__ == "__main__":

    #### Read config ####
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #### Connect to Ray and WandB ####
    head_node = str(os.environ["head_node"])
    port = str(os.environ["port"])
    print(f"Connecting to head node: {head_node}:{port}")
    ray.init(address=head_node + ":" + port)

    wandb.init(
        project=config["project"],
    )

    #### Set up sweep config ####
    meta_params = [
        "script",
        "bsub_W",
        "bsub_n",
        "bsub_M",
        "bsub_q",
        "bsub_R",
        "bsub_gpu",
        "h5ad",
        "out_dir",
        "save_adata",
        "save_model",
        "minimal",
        "train_datasets",
        "test_dataset",
        "train_combs",
        "exclude_combs",
        "eval_combs",
        "sweep_combs",
        "train_conds",
        "eval_conds",
        "sweep_conds",
        "plot_umap_combs",
        "plot_umap_conds",
        "plot_heatmap_combs",
        "plot_heatmap_conds",
        "latent_space",
        "n_dims_train",
        "n_dims_eval",
        "vae_eval_rep",
        "h5ad_vae_train",
        "vae_train_path",
        "h5ad_vae_eval",
        "vae_eval_path",
    ]
    model_params = [
        "iterations",
        "batch_size",
        "lr_init",
        "lr_schedule",
        "lr_b1",
        "lr_b2",
        "opt_n_steps",
        "n_src_cells",
        "n_src_cells_eval",
        "cond_embed_dim",
        "pooling",
        "pool_sample_covariates",
        "pool_num_heads",
        "pool_qkv_dim",
        "pool_dropout",
        "pool_act_fn",
        "pool_seed_dim",
        "pool_transformer_block",
        "pool_layer_norm",
        "cond_layers_before_pool",
        "cond_n_layers_before_pool",
        "cond_hidden_dim_before_pool",
        "cond_dropout_before_pool",
        "cond_layers_after_pool",
        "cond_n_layers_after_pool",
        "cond_hidden_dim_after_pool",
        "cond_dropout_after_pool",
        "cond_output_dropout",
        "cond_num_heads",
        "cond_transformer",
        "cond_layer_norm",
        "cond_act_fn",
        "ot_num_layers_t",
        "ot_num_layers_x",
        "ot_num_layers_decoder",
        "ot_t_hidden_dim",
        "ot_x_hidden_dim",
        "ot_t_embed_dim",
        "ot_x_embed_dim",
        "ot_joint_hidden_dim",
        "ot_act_fn",
        "ot_dropout_t",
        "ot_dropout_x",
        "ot_dropout_decoder",
        "ot_flow_type",
        "ot_flow_noise",
        "ot_eps",
        "ot_tau_a",
        "ot_tau_b",
        "ot_n_frequencies",
    ]
    sweep_params = {key: config[key] for key in meta_params}
    sweep_params.update({key: get_ray_param(config[key]) for key in model_params})

    #### Run tuning with Ray ####
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tuner = tune.Tuner(
        run_otfm_bsub,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            num_samples=config["num_trials"],
            metric=config["metric"],
            mode="min",
            max_concurrent_trials=config["max_concurrent_trials"],
        ),
        run_config=train.RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project=config["project"],
                    api_key=config["wandb_key"],
                    log_config=True,
                )
            ],
            storage_path=out_dir,
            name=config["project"],
        ),
        param_space=sweep_params,
    )
    results = tuner.fit()

    # Save results
    with open(f"{out_dir}/results.pkl", "wb") as f:
        cloudpickle.dump(results, f)

    print("Done. Shutting down ray cluster")
    ray.shutdown()
