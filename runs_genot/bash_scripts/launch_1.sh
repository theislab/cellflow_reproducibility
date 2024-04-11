#!/bin/bash
#SBATCH -J fmpert
#SBATCH -o fmpert-1.txt
#SBATCH -e fmpert-2.txt
#SBATCH -q gpu_normal
#SBATCH -t 24:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --nice=10000

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_genot/train_combosciplex.py +logger=wandb 