#!/bin/bash

#SBATCH -o o-chemcpa-combosciplex.out

#SBATCH -e o-chemcpa-combosciplex.err

#SBATCH -J o-chemcpa-combosciplex

#SBATCH -p cpu_p

#SBATCH --qos=cpu_priority

#SBATCH --mem=120G

#SBATCH -t 0-00:30:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/CPA_new/run_chemcpa.py ae_hparams=chemcpa generic_params=chemcpa launcher=cpu_normal logger=cpa
