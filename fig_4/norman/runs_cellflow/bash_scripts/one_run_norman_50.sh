#!/bin/bash

#SBATCH -o n-otfm-norman_%j.out

#SBATCH -e n-otfm-norman_%j.err

#SBATCH -J n-otfm-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow



python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_4/norman/runs_cellflow/train_norman_pca_50.py \
    dataset=norman \
    model=norman \
    training=norman \
    logger=norman
