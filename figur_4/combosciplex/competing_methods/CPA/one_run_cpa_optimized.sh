#!/bin/bash

#SBATCH -o o-cpa-combosciplex_2.out

#SBATCH -e o-cpa-combosciplex_2.err

#SBATCH -J o-cpa-combosciplex_2

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=110G

#SBATCH -t 0-08:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/CPA_new/run_cpa_optimized.py ae_hparams=cpa generic_params=cpa logger=cpa generic_params.split=2
