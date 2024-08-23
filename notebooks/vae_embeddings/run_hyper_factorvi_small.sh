#!/bin/bash

#SBATCH -o hyper_s_factorvi.out

#SBATCH -e hyper_s_factorvi.err

#SBATCH -J hyper_s_factorvi

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 0-12:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot_scvi

python /home/icb/dominik.klein/git_repos/ot_pert_new/notebooks/vae_embeddings/hyperparameter_factorvi_sciplex_small.py
