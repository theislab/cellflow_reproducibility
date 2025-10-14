#!/bin/bash

#SBATCH -o h_pbmc_%j.out

#SBATCH -e h_pbmc_%j.err

#SBATCH -J h_pbmc_%j

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_cytokine.py --multirun \
    dataset=pbmc_new_cytokine \
    model=pbmc_new_cytokine \
    training=pbmc_new_cytokine \
    logger=zebrafish \
    launcher=gpu_normal_large_mem \
    +hparams_search=pbmc_datasets \


