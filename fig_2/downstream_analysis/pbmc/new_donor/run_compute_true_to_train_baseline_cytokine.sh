#!/bin/bash
#SBATCH -o baseline_cyto.out
#SBATCH -e baseline_cyto.err
#SBATCH -J baseline_cyto
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python ./compute_true_to_train_baseline_cytokine.py