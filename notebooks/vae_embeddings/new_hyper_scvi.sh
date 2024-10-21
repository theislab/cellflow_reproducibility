#!/bin/bash

#SBATCH -o vae-norman_%j.out

#SBATCH -e vae-norman_%j.err

#SBATCH -J vae-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

echo "Start new_hyper_scvi.sh."

output_dir=${1}
adata_train=${2}
adata_val=${3}
adata_test=${4}

source /home/haicu/soeren.becker/.bashrc
conda activate /home/haicu/soeren.becker/miniconda3/envs/env_cfp

python new_hyper_scvi.py \
    --output_dir "${output_dir}" \
    --filename_adata_train "${adata_train}" \
    --filename_adata_val "${adata_val}" \
    --filename_adata_test "${adata_test}"

# /home/haicu/soeren.becker/repos/ot_pert_reproducibility/norman2019/norman_preprocessed_adata
# adata_train_pca_50_split_0.h5ad
# adata_val_pca_50_split_0.h5ad
# adata_test_pca_50_split_0.h5ad

echo "Finished new_hyper_scvi.sh."