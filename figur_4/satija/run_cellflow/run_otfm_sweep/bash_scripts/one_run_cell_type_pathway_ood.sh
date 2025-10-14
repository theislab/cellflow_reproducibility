#!/bin/bash

#SBATCH -o ./logs/otfm_one_satija_K562_INS.out

#SBATCH -e ./logs/otfm_one_satija_K562_INS.err

#SBATCH -J ./logs/otfm_one_satija_K562_INS

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=100G

#SBATCH -t 1-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc
conda activate ot_pert

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_cell_type_ood_split_A549_IFNB \
# logger=satija_cell_type_ood_A549_IFNB \
# training=satija_leaveout_best_BXCP3_IFNG \
# launcher=slurm_icb \
# model=satija_leaveout_best_BXCP3_IFNG 

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_cell_type_ood_split_BXPC3_IFNG \
# logger=satija_cell_type_ood_BXCP3_IFNG \
# training=satija_leaveout_best_BXCP3_IFNG \
# launcher=slurm_icb \
# model=satija_leaveout_best_BXCP3_IFNG 

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_cell_type_ood_split_HAP1_TGFB \
# logger=satija_cell_type_ood_HAP1_TGFB \
# training=satija_leaveout_best_BXCP3_IFNG \
# launcher=slurm_icb \
# model=satija_leaveout_best_BXCP3_IFNG 

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_cell_type_ood_split_HT29_TNFA \
# logger=satija_cell_type_ood_HT29_TNFA \
# training=satija_leaveout_best_BXCP3_IFNG \
# launcher=slurm_icb \
# model=satija_leaveout_best_BXCP3_IFNG 

python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
dataset=satija_cell_type_ood_split_K562_INS \
logger=satija_cell_type_ood_K562_INS \
training=satija_leaveout_best_BXCP3_IFNG \
launcher=slurm_icb \
model=satija_leaveout_best_BXCP3_IFNG 
