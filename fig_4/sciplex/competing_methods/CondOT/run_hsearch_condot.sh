#!/bin/bash

#SBATCH -o hsearch_condot.out

#SBATCH -e hsearch_condot.err

#SBATCH -J hsearch_condot

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 1-00:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/sciplex/CondOT/hsearch_condot.py
