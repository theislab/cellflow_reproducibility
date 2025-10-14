#!/bin/bash

#SBATCH -o ./logs/evaluation_gene_split_identity_BXPC3.out

#SBATCH -e ./logs/evaluation_gene_split_identity_BXPC3.err

#SBATCH -J evaluation_gene_split_identity_BXPC3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert
 
python ../compute_metrics_identity.py BXCP3 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_cell_type/BXPC3/ ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line
