#!/bin/bash

#SBATCH -o pbmc_new_c.out

#SBATCH -e pbmc_new_c.err

#SBATCH -J pbmc_new_c

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --mem=20G

#SBATCH -t 0-00:15:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_pbmc_new_cytokine.py --multirun dataset=pbmc_new_cytokine +hparams_search=pbmc_new_cytokine model=pbmc_new_cytokine training=pbmc_new_cytokine logger=zebrafish launcher=gpu_normal_large_mem
