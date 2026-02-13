#!/bin/bash

#SBATCH -o pbmc_eps.out

#SBATCH -e pbmc_eps.err

#SBATCH -J pbmc_eps

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --mem=20G

#SBATCH -t 0-00:15:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py --multirun dataset=pbmc_new_donor +hparams_search=pbmc_new_donor_ablation_batch_size model=pbmc_new_donor training=pbmc_new_donor_ablation_batch_size logger=zebrafish launcher=gpu_normal_large_mem
