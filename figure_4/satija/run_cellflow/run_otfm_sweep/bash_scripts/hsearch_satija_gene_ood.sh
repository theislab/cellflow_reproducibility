#!/bin/bash

#SBATCH -o h-otfm-satija-ood_split.out

#SBATCH -e h-otfm-satija-ood_split.err

#SBATCH -J h-otfm-satija-ood_split

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 2-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc
conda activate ot_pert

# python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py \
#  --multirun \
#  dataset=satija_ifng \
#  hparams_search=satija_ifng model=satija_ifng \
#  training=satija_ifng \
#  logger=satija_ifng #launcher=slurm_icb


python /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/run_otfm_sweep/train_satija_gene_ood.py --m dataset=satija_gene_ood_split_0 logger=satija_gene_ood_split_0 training=satija_ifng launcher=slurm_icb model=satija_leaveout hparams_search=satija_ifng 

