#!/bin/bash

#SBATCH -o ./logs/otfm_one_satija_ood_split_3.out

#SBATCH -e ./logs/otfm_one_satija_ood_split_3.err

#SBATCH -J ./logs/otfm_one_satija_ood_split_3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=100G

#SBATCH -t 1-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc
conda activate ot_pert

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_gene_ood_split_0 \
# logger=satija_gene_ood_split_0 \
# training=satija_ifng \
# launcher=slurm_icb \
# model=satija_leaveout_best

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_gene_ood_split_1 \
# logger=satija_gene_ood_split_1 \
# training=satija_ifng \
# launcher=slurm_icb \
# model=satija_leaveout_best

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
# dataset=satija_gene_ood_split_2 \
# logger=satija_gene_ood_split_2 \
# training=satija_ifng \
# launcher=slurm_icb \
# model=satija_leaveout_best

python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
dataset=satija_gene_ood_split_3 \
logger=satija_gene_ood_split_3 \
training=satija_ifng \
launcher=slurm_icb \
model=satija_leaveout_best
