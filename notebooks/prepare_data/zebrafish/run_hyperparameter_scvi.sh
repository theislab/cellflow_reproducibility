#!/bin/bash

#SBATCH -o hyper_scvi.out

#SBATCH -e hyper_scvi.err

#SBATCH -J hyper_scvi

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 4-00:00:00

#SBATCH --nice=1


source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/notebooks/prepare_data/zebrafish/hyperparameter_scvi.py
