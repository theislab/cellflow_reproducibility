#!/bin/bash

#SBATCH -o 3pbmc_%j.out

#SBATCH -e 3pbmc_%j.err

#SBATCH -J pbmc_%j

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --constraint=a100_80gb|h100_80gb

#SBATCH --mem=400G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_cytokine2.py dataset=pbmc_new_cytokine model=pbmc_new_cytokine training=pbmc_new_cytokine launcher=gpu_normal_large_mem
