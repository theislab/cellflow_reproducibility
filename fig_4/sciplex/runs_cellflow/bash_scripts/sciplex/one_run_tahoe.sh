#!/bin/bash

#SBATCH -o 3tahoe_one_sciplex.out

#SBATCH -e 3tahoe_one_sciplex.err

#SBATCH -J 3tahoe_one_sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=500G

#SBATCH -t 0-24:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_4/sciplex/runs_cellflow/train_tahoe.py dataset=tahoe logger=sciplex training=tahoe launcher=gpu_normal model=tahoe
