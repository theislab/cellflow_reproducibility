#!/bin/bash
# BSUB -J BASELINES
# BSUB -W 48:00
# BSUB -n 1
# BSUB -M 50G
# BSUB -q long
# BSUB -R "span[hosts=1]"
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -o /home/fleckj/logs/BASELINES.out
# BSUB -e /home/fleckj/logs/BASELINES.err

ml Miniforge3
conda activate py310_cf_stable

python /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/baselines/compute_baselines.py