#!/bin/bash

#SBATCH -o eval_model.out

#SBATCH -e eval_model.err

#SBATCH -J eval_model

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 0-24:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/load_models/sciplex/pca_attention_pooling/evaluate_sciplex_1.py
