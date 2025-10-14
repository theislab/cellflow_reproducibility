#!/bin/bash

#SBATCH -o h-otfm-norman_%j.out

#SBATCH -e h-otfm-norman_%j.err

#SBATCH -J h-otfm-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow



python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_4/norman/runs_cellflow/train_norman_pca_50_gene_space.py --multirun \
    dataset=norman \
    launcher=gpu_long \
    model=norman \
    training=norman \
    logger=norman +hparams_search=hparams_norman_gene_space
