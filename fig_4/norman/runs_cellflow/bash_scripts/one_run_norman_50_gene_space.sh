#!/bin/bash

#SBATCH -o g-otfm-norman_%j.out

#SBATCH -e g-otfm-norman_%j.err

#SBATCH -J g-otfm-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow



python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_4/norman/runs_cellflow/train_norman_pca_50_gene_space.py \
    dataset=norman_gene_space \
    model=norman \
    training=norman \
    logger=norman
