import sys
import os
import warnings
import yaml

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # load config
    config_filename = sys.argv[1]
    with open(config_filename, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    script = config.pop("script")
    bsub_params = {
        key.split("_")[1]: config.pop(key)
        for key in list(config.keys())
        if key.startswith("bsub")
    }

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
    n_splits = config.pop("n_splits")
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
                raise ValueError(
                    f"Length of {split_param} must be equal to 1 or `n_splits`."
                )
        else:
            split_params[split_param] = [split_param_list for _ in range(n_splits)]

    # run OT-FM
    out_prefix = config["out_prefix"]
    out_dir = config["out_dir"]
    for i in range(n_splits):
        config.update({key: val[i] for key, val in split_params.items()})
        if n_splits > 1:
            split = f"split{i+1}"
            config["out_dir"] = os.path.join(out_dir, split)
            config["out_prefix"] = f"{out_prefix}_{split}" if out_prefix else split

        os.makedirs(config["out_dir"], exist_ok=True)
        run_config = os.path.join(config["out_dir"], "config.yml")
        with open(run_config, "w") as f:
            f.write(yaml.dump(config, sort_keys=False))

        bsub_params["J"] = f"{config['out_prefix']}"
        cmd = ["bsub"]
        cmd.extend([f'-{key} "{str(value)}"' for key, value in bsub_params.items()])
        cmd.extend(
            ["-o", os.path.join(config["out_dir"], f"{config['out_prefix']}.out")]
        )
        cmd.extend(
            ["-e", os.path.join(config["out_dir"], f"{config['out_prefix']}.err")]
        )
        cmd.extend(["python", script])
        cmd.extend([run_config])
        cmd = " ".join(cmd)
        os.system(cmd)
