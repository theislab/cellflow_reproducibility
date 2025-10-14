#!/bin/bash
# BSUB -J TRAIN_EVAL
# BSUB -W 10:00
# BSUB -n 1
# BSUB -M 32G
# BSUB -q long
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -o logs/TRAIN_EVAL_%J.out


ml Miniforge3
conda activate py310_cf_stable

echo "NAME: $CF_NAME"
echo "TRIAL_NAME: $CF_TRIAL_NAME"
echo "CONFIG: $CF_CONFIG_PATH"
echo "TRAIN: $CF_TRAIN_PATH"
echo "TEST: $CF_TEST_PATH"


python /home/fleckj/projects/cellflow/scripts/runs/run_cf_train_eval.py \
    --config $CF_CONFIG_PATH \
    --train_adata $CF_TRAIN_PATH \
    --test_adata $CF_TEST_PATH \
    --name $CF_NAME \
    --trial_name $CF_TRIAL_NAME
