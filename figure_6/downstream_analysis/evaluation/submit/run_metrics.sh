#!/bin/bash
# BSUB -J METRICS
# BSUB -W 10:00
# BSUB -n 1
# BSUB -M 32G
# BSUB -q long
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -o logs/METRICS_%J.out


ml Miniforge3
conda activate py310_cf_stable

echo "NAME: $CF_NAME"
echo "TRIAL_NAME: $CF_TRIAL_NAME"
echo "ADATA: $CF_ADATA_PATH"


python /home/fleckj/projects/cellflow/scripts/runs/run_additional_metrics.py \
    --full_adata $CF_ADATA_PATH \
    --name $CF_NAME \
    --trial_name $CF_TRIAL_NAME
