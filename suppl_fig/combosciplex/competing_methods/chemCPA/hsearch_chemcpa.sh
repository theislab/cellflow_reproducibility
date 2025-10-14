#!/bin/bash

#SBATCH -o h-chemcpa-combosciplex.out

#SBATCH -e h-chemcpa-combosciplex.err

#SBATCH -J h-chemcpa-combosciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --mem=20G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/CPA_new/run_chemcpa.py --multirun ae_hparams=chemcpa generic_params=chemcpa launcher=gpu_long logger=cpa +hparams_search=hparams_combosciplex_chemcpa
