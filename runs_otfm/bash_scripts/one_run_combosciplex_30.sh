#!/bin/bash

#SBATCH -o otfm_one_csciplex.out

#SBATCH -e otfm_one_csciplex.err

#SBATCH -J otfm_one_csciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/train_combosciplex.py dataset=combosciplex_30 model=combosciplex training=training_combosciplex
