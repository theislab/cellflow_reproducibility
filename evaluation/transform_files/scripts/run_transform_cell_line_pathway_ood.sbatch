#!/bin/bash

#SBATCH -o ./logs/transform_K562_INS.out

#SBATCH -e ./logs/transform_K562_INS.err

#SBATCH -J transform_K562_INS

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert

cd /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/evaluation/transform_files
 
# python save_transformed_files.py A549_IFNB /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/A549_IFNB /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line_pathway ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line_pathway

# python save_transformed_files.py BXPC3_IFNG /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/BXPC3_IFNG /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line_pathway ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line_pathway

# python save_transformed_files.py HAP1_TGFB /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/HAP1_TGFB /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line_pathway ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line_pathway

# python save_transformed_files.py HT29_TNFA /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/HT29_TNFA /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line_pathway ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line_pathway

python save_transformed_files.py K562_INS /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_pathway_cell_type/K562_INS /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line_pathway ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line_pathway