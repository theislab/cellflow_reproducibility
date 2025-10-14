#!/bin/bash

#SBATCH -o closest_emb_sciplex.out
#SBATCH -e closest_emb_sciplex.err
#SBATCH -J closest_emb_sciplex
#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=200G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow


# Run your program with the tuple as an argument
python closest_embedding.py 
