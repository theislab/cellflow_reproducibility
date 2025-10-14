#!/bin/bash

#SBATCH -o closest_embedding_norman.out

#SBATCH -e closest_embedding_norman.err

#SBATCH -J closest_embedding_norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=100G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python closest_embedding.py