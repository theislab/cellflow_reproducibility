#!/bin/bash

#SBATCH -o otfm_one_combosciplex.out

#SBATCH -e otfm_one_combosciplex.err

#SBATCH -J otfm_one_combosciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/train_combosciplex.py dataset=combosciplex logger=combosciplex training=combosciplex launcher=slurm_icb model=combosciplex dataset.split=1
