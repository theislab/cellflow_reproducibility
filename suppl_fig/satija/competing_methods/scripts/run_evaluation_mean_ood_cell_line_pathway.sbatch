#!/bin/bash

#SBATCH -o ./logs/evaluation_mean_K562_INS.out

#SBATCH -e ./logs/evaluation_mean_K562_INS.err

#SBATCH -J evaluation_mean_K562_INS

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert
 
# python ../compute_metrics_mean_pathway_cell_line.py A549_IFNB /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/A549_IFNB ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line_pathway /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits.h5ad

# python ../compute_metrics_mean_pathway_cell_line.py BXPC3_IFNG /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/BXPC3_IFNG ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line_pathway /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits.h5ad

# python ../compute_metrics_mean_pathway_cell_line.py HAP1_TGFB /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/HAP1_TGFB ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line_pathway /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits.h5ad

# python ../compute_metrics_mean_pathway_cell_line.py HT29_TNFA /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/HT29_TNFA ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line_pathway /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits.h5ad

python ../compute_metrics_mean_pathway_cell_line.py K562_INS /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/K562_INS ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_cell_line_pathway /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits.h5ad