#!/bin/bash

#SBATCH -o ./logs/otfm_one_satija_ood_split_0.out

#SBATCH -e ./logs/otfm_one_satija_ood_split_0.err

#SBATCH -J ./logs/otfm_one_satija_ood_split_0

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=100G

#SBATCH -t 1-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc
conda activate ot_pert

python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
dataset=satija_cell_type_ood_split_BXPC3 \
logger=satija_cell_type_ood_BXCP3 \
training=satija_leaveout_best_BXCP3 \
launcher=slurm_icb \
model=satija_leaveout_best_BXCP3 
