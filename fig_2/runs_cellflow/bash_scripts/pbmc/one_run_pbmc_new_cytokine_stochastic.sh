#!/bin/bash

#SBATCH -o lpbmc_stochastic.out

#SBATCH -e lpbmc_stochastic.err

#SBATCH -J lpbmc_stochastic
#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint="a100_80gb"

#SBATCH --gres=gpu:1

#SBATCH --mem=500G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_cytokine_stochastic.py dataset=pbmc_new_cytokine_stochastic model=pbmc_new_cytokine_stochastic training=pbmc_new_cytokine_stochastic logger=zebrafish training.num_iterations=1000
