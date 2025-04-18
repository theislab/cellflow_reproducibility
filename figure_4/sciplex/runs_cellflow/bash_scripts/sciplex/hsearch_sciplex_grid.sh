#!/bin/bash

#SBATCH -o h-otfm-sciplex.out

#SBATCH -e h-otfm-sciplex.err

#SBATCH -J h-otfm-sciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_max

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_sciplex.py --multirun dataset=sciplex +hparams_search=hparams_sciplex_grid model=sciplex training=sciplex logger=sciplex launcher=gpu_max
