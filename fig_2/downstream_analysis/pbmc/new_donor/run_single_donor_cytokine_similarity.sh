#!/bin/bash
#SBATCH -o logs/baseline_cyto_%x.out
#SBATCH -e logs/baseline_cyto_%x.err
#SBATCH -J baseline_cyto
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load environment
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

# Run the script with donor argument
python ./compute_true_to_train_baseline_cytokine.py "$1"
