#!/bin/bash

#SBATCH -o ./logs/transform_3.out

#SBATCH -e ./logs/transform_3.err

#SBATCH -J transform_3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert
 
# python ../save_transformed_files.py split_0 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_genes/ ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_genes

# python ../save_transformed_files.py split_1 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_genes/ ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_genes

# python ../save_transformed_files.py split_2 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_genes/ ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_genes

python ../save_transformed_files.py split_3 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/best_ckpts/sweep_genes/ ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/generated_data/sweep_genes
