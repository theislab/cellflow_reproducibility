#!/bin/bash

#SBATCH -o otfm_one_combosciplex1.out

#SBATCH -e otfm_one_combosciplex1.err

#SBATCH -J otfm_one_combosciplex1

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_combosciplex_optimized.py dataset=combosciplex logger=combosciplex training=combosciplex launcher=gpu_normal model=combosciplex dataset.split=1
