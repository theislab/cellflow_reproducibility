#!/bin/bash

#SBATCH -o h-genot-sciplex.out

#SBATCH -e h-genot-sciplex.err

#SBATCH -J h-genot-sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_genot/train_sciplex.py --multirun dataset=sciplex +hparams_search=hparams_sciplex launcher=slurm_icb_long model=sciplex training=sciplex logger=sciplex
