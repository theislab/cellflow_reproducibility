#!/bin/bash

#SBATCH -o gears_one_satija.out

#SBATCH -e gears_one_satija.err

#SBATCH -J gears_one_satija

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint=a100_40gb|a100_80gb

#SBATCH --gres=gpu:1

#SBATCH --mem=90G

#SBATCH -t 00:10:00

#SBATCH --nice=0

# source ${HOME}/.bashrc_new
source /lustre/groups/ml01/workspace/lea.zimmermann/software/miniconda3/etc/profile.d/conda.sh
conda activate gears

python /home/icb/lea.zimmermann/projects/pertot/ot_pert_reproducibility/runs_gears/train_satija_ifng_BXPC3.py