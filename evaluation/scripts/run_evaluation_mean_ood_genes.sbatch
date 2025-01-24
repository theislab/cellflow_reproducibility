#!/bin/bash

#SBATCH -o ./logs/evaluation_gene_split_mean_0.out

#SBATCH -e ./logs/evaluation_gene_split_mean_0.err

#SBATCH -J evaluation_gene_split_mean_0

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert

cd /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/evaluation/evaluate_baselines

python compute_metrics_mean_genes.py split_0 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_gene/split0 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits_gene_ood.h5ad
 
# python compute_metrics_mean_genes.py split_1 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_gene/split1 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits_gene_ood.h5ad

# python compute_metrics_mean_genes.py split_2 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_gene/split2 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits_gene_ood.h5ad

# python compute_metrics_mean_genes.py split_3 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/adata_ood_final_genesIFNG_IFNB_TNFA_TGFB_INS_hvg-500_pca-100_counts_ms_0.5_3 ood_genes /lustre/groups/ml01/workspace/alessandro.palma/ot_pert/out/results_metrics/metrics/sweep_gene/split3 /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/reference/full_adata_with_splits_gene_ood.h5ad
