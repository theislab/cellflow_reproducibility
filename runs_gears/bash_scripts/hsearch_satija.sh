#!/bin/bash

#SBATCH -o h-gears-satija.out

#SBATCH -e h-gears-satija.err

#SBATCH -J h-gears-satija

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc_new
source /lustre/groups/ml01/workspace/lea.zimmermann/software/miniconda3/etc/profile.d/conda.sh
conda activate gears

python /home/icb/lea.zimmermann/projects/pertot/ot_pert_reproducibility/runs_gears/train_satija_ifng_BXPC3.py --multirun dataset=satija_ifng_bxpc3 hparams_search=hparams_satija model=satija training=training_satija logger=satija_ifng
