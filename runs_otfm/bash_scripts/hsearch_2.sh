#!/bin/bash

#SBATCH -o hsearch2.out

#SBATCH -e hsearch2.err

#SBATCH -J hsearch2

#SBATCH -p gpu_p

#SBATCH --qos=gpu_short

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/train_combosciplex.py --multirun hydra/launcher=submitit_slurm