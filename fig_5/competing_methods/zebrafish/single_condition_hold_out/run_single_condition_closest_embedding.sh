#!/bin/bash

#SBATCH -o 2zebra_closest_emb.out
#SBATCH -e 2zebra_closest_emb.err
#SBATCH -J 2zebra_closest_emb
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=h100_80gb
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python single_condition_closest_embedding.py
