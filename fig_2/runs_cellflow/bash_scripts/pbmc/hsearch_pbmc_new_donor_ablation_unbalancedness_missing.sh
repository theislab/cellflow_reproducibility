#!/bin/bash

#SBATCH -o m_pbmc_unb.out

#SBATCH -e m_pbmc_unb.err

#SBATCH -J m_pbmc_unb

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --mem=20G

#SBATCH -t 0-00:15:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py --multirun dataset=pbmc_new_donor +hparams_search=pbmc_new_donor_ablation_unbalancedness_missing model=pbmc_new_donor training=pbmc_new_donor_ablation_unbalancedness logger=zebrafish launcher=gpu_normal_large_mem
