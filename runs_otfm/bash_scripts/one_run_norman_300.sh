#!/bin/bash

#SBATCH -o h-otfm-norman_%j.out

#SBATCH -e h-otfm-norman_%j.err

#SBATCH -J h-otfm-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source /home/haicu/soeren.becker/.bashrc
conda activate /home/haicu/soeren.becker/miniconda3/envs/env_cfp



python /home/haicu/soeren.becker/repos/ot_pert_reproducibility/runs_otfm/train_norman_pca_300.py \
    dataset=norman \
    +hparams_search=hparams_norman \
    model=norman \
    training=norman \
    logger=norman
