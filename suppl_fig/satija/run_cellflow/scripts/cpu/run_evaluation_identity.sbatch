#!/bin/bash

#SBATCH -o ./logs/evaluation_gene_split_identity_0.out

#SBATCH -e ./logs/evaluation_gene_split_identity_0.err

#SBATCH -J evaluation_gene_split_identity_0

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=300G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert

cd /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/evaluation/evaluate_baselines
 
python compute_metrics_identity.py 0 
# python compute_metrics_identity.py 1
# python compute_metrics_identity.py 2