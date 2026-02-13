#!/bin/bash

#SBATCH -o pbmc_stochastic.out

#SBATCH -e pbmc_stochastic.err

#SBATCH -J pbmc_stochastic
#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint="a100_80gb"

#SBATCH --gres=gpu:1

#SBATCH --mem=500G

#SBATCH -t 0-24:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py dataset=pbmc_new_donor  model=pbmc_new_donor training=pbmc_new_donor_ablation_unbalancedness logger=zebrafish launcher=gpu_normal_large_mem
