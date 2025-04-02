#!/bin/bash

#SBATCH -o h-cpa-combosciplex.out

#SBATCH -e h-cpa-combosciplex.err

#SBATCH -J h-cpa-combosciplex

#SBATCH -p cpu_p

#SBATCH --qos=cpu_short

#SBATCH --mem=20G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/groups/ml01/workspace/ot_pert_new/conda_envs/ot_pert_cpa

python /home/icb/dominik.klein/git_repos/ot_pert_new/competing_methods/combosciplex/CPA_new/run_cpa.py --multirun ae_hparams=cpa generic_params=cpa launcher=gpu_normal logger=cpa +hparams_search=hparams_combosciplex_cpa
