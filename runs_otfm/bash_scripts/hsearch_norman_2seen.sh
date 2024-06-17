#!/bin/bash
#SBATCH -J norman2seen
#SBATCH -o norman-2seen-o-%j.txt
#SBATCH -e norman-2seen-e-%j.txt
#SBATCH -q gpu_long
#SBATCH -t 96:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160GB
#SBATCH --nice=10000

source ${HOME}/.bashrc
mamba activate jax

python /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_otfm/train_norman.py --multirun dataset=norman_2seen
