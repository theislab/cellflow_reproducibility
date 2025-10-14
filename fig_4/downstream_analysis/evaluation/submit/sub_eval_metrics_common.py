import os

os.chdir("/home/fleckj/")

SWEEP_PATH = "/home/fleckj/projects/cellflow/results/psweep_organoids_common_v2/"
RUN_NAME = "cellflow_0a37dcb9"
RUN_CONFIG = f"{SWEEP_PATH}/{RUN_NAME}/params.pkl"

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
            adata_path = f"{DATA_PATH}/{task}/{ds}/{holdout}/adata_full.h5ad"
            SPLIT_DATA[split_name] = adata_path


for split_name, split_data in SPLIT_DATA.items():
    THIS_ENV = f"all, CF_NAME={split_name}, CF_ADATA_PATH={split_data}, CF_TRIAL_NAME={RUN_NAME}"
    os.system(
        f"bsub -env '{THIS_ENV}' < /home/fleckj/projects/cellflow/scripts/runs/submit/run_metrics.sh"
    )

# # Just run py script
# for split_name, split_data in SPLIT_DATA.items():
#     os.system(
#         f"python /home/fleckj/projects/cellflow/scripts/runs/run_all_metrics.py --full_adata {split_data} --name {split_name} --trial_name {RUN_NAME}"
#     )
