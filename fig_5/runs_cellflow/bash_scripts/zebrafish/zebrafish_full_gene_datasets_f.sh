#!/bin/bash

#SBATCH -o fullg-zebra.out

#SBATCH -e fullg-zebra.err

#SBATCH -J fullg-zebra

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_zebrafish_full_gene_f.py --multirun dataset=zebrafish_full_gene +hparams_search=zebrafish_full_gene_datasets model=zebrafish_single_condition training=zebrafish_full_gene logger=zebrafish launcher=gpu_normal
