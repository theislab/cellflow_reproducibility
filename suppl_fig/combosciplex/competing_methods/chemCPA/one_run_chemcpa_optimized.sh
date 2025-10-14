#!/bin/bash

#SBATCH -o o-chemcpa-combosciplex_3.out

#SBATCH -e o-chemcpa-combosciplex_3.err

#SBATCH -J o-chemcpa-combosciplex_3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=110G

#SBATCH -t 0-08:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/CPA_new/run_chemcpa_optimized.py ae_hparams=chemcpa generic_params=chemcpa trainer_hparams=chemcpa logger=cpa generic_params.split=3
