#!/bin/bash

#SBATCH -o ./logs/launch_gene_split_0.out

#SBATCH -e ./logs/launch_gene_split_0.err

#SBATCH -J launch_gene_split_0

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=300G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate ot_pert 

# python ../generate_dataset_gene_split.py --ood_split split_0 --hvg 500 --pca_dim 100 --ms 0.5 --split_path /lustre/groups/ml01/workspace/ot_perturbation/data/satija/datasets/adata_ood_final_genes/gene_split.csv 
# python ../generate_dataset_gene_split.py --ood_split split_1 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_gene_split.py --ood_split split_2 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_gene_split.py --ood_split split_3 --hvg 500 --pca_dim 100 --ms 0.5