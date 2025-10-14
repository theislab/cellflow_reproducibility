#!/bin/bash

#SBATCH -o cns_first_t_zebrafish.out

#SBATCH -e cns_first_t_zebrafish.err

#SBATCH -J cns_first_t_zebrafish

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 0-24:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_zebrafish_first_t_cns.py dataset=zebrafish_first_t logger=zebrafish training=zebrafish_single_condition model=zebrafish_single_condition
