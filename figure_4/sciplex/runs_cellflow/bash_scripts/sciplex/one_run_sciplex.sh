#!/bin/bash

#SBATCH -o otfm_one_sciplex.out

#SBATCH -e otfm_one_sciplex.err

#SBATCH -J otfm_one_sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_max

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_sciplex.py dataset=sciplex logger=sciplex training=sciplex launcher=gpu_normal model=sciplex
