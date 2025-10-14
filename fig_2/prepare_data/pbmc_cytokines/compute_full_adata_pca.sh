#!/bin/bash

#SBATCH -o compute_full_adata_pca.out
#SBATCH -e compute_full_adata_pca.err
#SBATCH -J compute_full_adata_pca
#SBATCH -p cpu_p
#SBATCH --qos=cpu_long
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp
python compute_full_adata_pca.py "$split"
