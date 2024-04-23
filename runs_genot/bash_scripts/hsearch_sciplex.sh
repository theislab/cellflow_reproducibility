#!/bin/bash

#SBATCH -o h-sciplex.out

#SBATCH -e h-sciplex.err

#SBATCH -J h-sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_genot/train_sciplex.py --multirun dataset=sciplex_biolord_split hparams_search=hparams_sciplex model=sciplex training=training_sciplex logger=sciplex
