#!/bin/bash

#SBATCH -o pbmc_new_d.out

#SBATCH -e pbmc_new_d.err

#SBATCH -J pbmc_new_d

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --mem=20G

#SBATCH -t 0-00:15:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_pbmc_new_donor.py --multirun dataset=pbmc_new_donor +hparams_search=pbmc_new_donor model=pbmc_new_donor training=pbmc_new_donor logger=zebrafish launcher=gpu_long_large_mem
