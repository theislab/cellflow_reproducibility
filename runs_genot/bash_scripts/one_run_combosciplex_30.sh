#!/bin/bash

#SBATCH -o one_combosciplex.out

#SBATCH -e one_combosciplex.err

#SBATCH -J one_combosciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint=a100_80gb

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 1-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_genot/train_combosciplex.py dataset=combosciplex model=combosciplex training=training_combosciplex logger=combosciplex
