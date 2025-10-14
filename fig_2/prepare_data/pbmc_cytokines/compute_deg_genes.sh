#!/bin/bash

#SBATCH -o compute_deg_genes.out
#SBATCH -e compute_deg_genes.err
#SBATCH -J compute_deg_genes
#SBATCH -p cpu_p
#SBATCH --qos=cpu_long
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp
python compute_deg_genes.py 