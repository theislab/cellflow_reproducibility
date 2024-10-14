#!/bin/bash

#SBATCH -o chemcpa_comb.out

#SBATCH -e chemcpa_comb.err

#SBATCH -J chemcpa_comb

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 4-00:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/chemCPA/hsearch_chemCPA.py
