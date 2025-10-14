#!/bin/bash

#SBATCH -o hr-cf-sciplex.out

#SBATCH -e hr-cf-sciplex.err

#SBATCH -J hr-cf-sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_4/sciplex/runs_cellflow/train_sciplex.py --multirun dataset=sciplex +hparams_search=hparams_sciplex_random model=sciplex training=sciplex logger=sciplex launcher=gpu_normal
