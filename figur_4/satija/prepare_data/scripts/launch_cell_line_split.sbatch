#!/bin/bash

#SBATCH -o ./logs/launch_path_cellt_split_0.out

#SBATCH -e ./logs/launch_path_cellt_split_0.err

#SBATCH -J launch_path_cellt_split_0

#SBATCH -J launch_gene_split_0

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=300G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate ot_pert

# python ../generate_dataset_cell_type_split.py --ood_cell_type BXPC3 --hvg 500 --pca_dim 100 --ms 0.5