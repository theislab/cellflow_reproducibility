#!/bin/bash

#SBATCH -o 4sciplex-h.out

#SBATCH -e 4sciplex-h.err

#SBATCH -J 4sciplex-h

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/train_sciplex_fast.py --multirun hydra/launcher=submitit_slurm dataset=sciplex_biolord_split hparams_search=hparams_sciplex logger=otfm_sciplex model=default_sciplex training=training_sciplex
