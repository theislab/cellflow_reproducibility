#!/bin/bash

#SBATCH -o h_search_pbmc_new_cytokine_stochastic.out

#SBATCH -e h_search_pbmc_new_cytokine_stochastic.err

#SBATCH -J h_search_pbmc_new_cytokine_stochastic

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint="a100_80gb|a100_40gb"

#SBATCH --gres=gpu:1

#SBATCH --mem=20G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_cytokine_stochastic.py --multirun dataset=pbmc_new_cytokine_stochastic +hparams_search=pbmc_new_cytokine_stochastic model=pbmc_new_cytokine_stochastic training=pbmc_new_cytokine_stochastic logger=zebrafish launcher=gpu_normal_large_mem
