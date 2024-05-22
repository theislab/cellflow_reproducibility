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

source ${HOME}/.bashrc
mamba activate jax

python /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_otfm/train_norman.py