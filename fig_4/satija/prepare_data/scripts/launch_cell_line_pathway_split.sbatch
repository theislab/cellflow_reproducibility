#!/bin/bash

#SBATCH -o ./logs/launch_path_cellt_split_ins.out

#SBATCH -e ./logs/launch_path_cellt_split_ins.err

#SBATCH -J launch_path_cellt_split_ins

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=300G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate ot_pert

# python ../generate_dataset_cell_type_pathway_split.py --ood_pathway IFNB --ood_cell_type A549 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_cell_type_pathway_split.py --ood_pathway IFNG --ood_cell_type BXPC3 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_cell_type_pathway_split.py --ood_pathway TGFB --ood_cell_type HAP1 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_cell_type_pathway_split.py --ood_pathway TNFA --ood_cell_type HT29 --hvg 500 --pca_dim 100 --ms 0.5
# python ../generate_dataset_cell_type_pathway_split.py --ood_pathway INS --ood_cell_type K562 --hvg 500 --pca_dim 100 --ms 0.5