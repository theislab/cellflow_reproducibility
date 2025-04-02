#!/bin/bash

#SBATCH -o interpolate_cdx4_cdx1a.out

#SBATCH -e interpolate_cdx4_cdx1a.err

#SBATCH -J interpolate_cdx4_cdx1a

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=180G

#SBATCH -t 2-00:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python interpolate_cdx4_cdx1a_cns.py