#!/bin/bash
#SBATCH -J fmpert
#SBATCH -o fmpert-o-%j.txt
#SBATCH -e fmpert-e-%j.txt
#SBATCH -q gpu_long
##BATCH --reservation=test
##SBATCH --qos=gpu_reservation
##SBATCH --reservation=h100
#SBATCH -t 48:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=240GB
#SBATCH --nice=10000

source ${HOME}/.bashrc
mamba activate jax

python -u /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_genot/train_zebrafish.py 