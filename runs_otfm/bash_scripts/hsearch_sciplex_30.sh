#!/bin/bash

#SBATCH -o 3sciplex-h.out

#SBATCH -e 3sciplex-h.err

#SBATCH -J 3sciplex-h

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/ot_pert_genot

python /home/icb/dominik.klein/git_repos/ot_pert_new/runs_otfm/train_sciplex.py --multirun hydra/launcher=submitit_slurm launcher=slurm_icb_cpu dataset=sciplex_30 hparams_search=hparams_sciplex logger=otfm_sciplex model=default_sciplex training=training_sciplex
