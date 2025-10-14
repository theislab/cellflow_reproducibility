#!/bin/bash

#SBATCH -o ./logs/transform_BXPC3.out

#SBATCH -e ./logs/transform_BXPC3.err

#SBATCH -J transform_BXPC3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert

python ../save_transformed_files.py BXCP3 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_cell_type/BXPC3 /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_cell_line/ ood_cell_line /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_cell_line