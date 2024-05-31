#!/bin/bash
#SBATCH -J fmpert
#SBATCH -o fmpert-o-%j.txt
#SBATCH -e fmpert-e-%j.txt
#SBATCH -q gpu_long
#SBATCH -t 96:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160GB
#SBATCH --nice=10000

source ${HOME}/.bashrc
mamba activate jax

python /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_otfm/train_norman.py --multirun --dataset=norman_1seen