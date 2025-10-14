import os

os.chdir("/home/fleckj/")

SWEEP_PATH = "/home/fleckj/projects/cellflow/results/psweep_organoids_common_v2/"
RUN_NAMES = ["cellflow_0a37dcb9"]

CONFIG_PATH = "/home/fleckj/projects/cellflow/results/ray_results/"
all_configs = [f"{path}/params.pkl" for path, _, _ in os.walk(CONFIG_PATH)]
original_run_config = {
    run_name: [config for config in all_configs if run_name in config][0]
    for run_name in RUN_NAMES
}

# Copy configs to correct directory
RUN_CONFIGS = {}
for run_name, run_config in original_run_config.items():
    os.system(f"cp {run_config} {SWEEP_PATH}/{run_name}/")
    RUN_CONFIGS[run_name] = f"{SWEEP_PATH}/{run_name}/params.pkl"


DATA_PATH = "/home/fleckj/projects/cellflow/data/datasets/organoids_combined/v8"
SPLIT_DATA = {}
split_tasks = ["combination", "transfer"]
exclude_patterns = ["DS_Store"]
for task in split_tasks:
    if not os.path.isdir(f"{DATA_PATH}/{task}"):
        continue
    split_datasets = os.listdir(f"{DATA_PATH}/{task}")
    split_datasets = [
        ds for ds in split_datasets if not any([p in ds for p in exclude_patterns])
    ]
    for ds in split_datasets:
        split_holdouts = os.listdir(f"{DATA_PATH}/{task}/{ds}")
        split_holdouts = [
            holdout
            for holdout in split_holdouts
            if not any([p in holdout for p in exclude_patterns])
        ]
        for holdout in split_holdouts:
            split_name = f"{task}_{ds}_{holdout}"
            train_path = f"{DATA_PATH}/{task}/{ds}/{holdout}/adata_train.h5ad"
            test_path = f"{DATA_PATH}/{task}/{ds}/{holdout}/adata_test.h5ad"
            SPLIT_DATA[split_name] = {"train": train_path, "test": test_path}


for run_name, run_config in RUN_CONFIGS.items():
    for split_name, split_data in SPLIT_DATA.items():
        THIS_ENV = f"all, CF_NAME={split_name}, CF_TRAIN_PATH={split_data['train']}, CF_TEST_PATH={split_data['test']}, CF_CONFIG_PATH={run_config}, CF_TRIAL_NAME={run_name}"
        os.system(
            f"bsub -env '{THIS_ENV}' < /home/fleckj/projects/cellflow/scripts/runs/submit/run_train_eval.sh"
        )
