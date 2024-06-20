#!/bin/bash

#SBATCH -J norman1seen
#SBATCH -o fmpert-1seen-o-%j.txt
#SBATCH -e fmpert-1seen-e-%j.txt
#SBATCH -q gpu_long
#SBATCH -t 96:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160GB
#SBATCH --nice=10000

source ${HOME}/.bashrc
mamba activate jax

python /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_otfm/train_norman.py --multirun dataset=norman_1seen
