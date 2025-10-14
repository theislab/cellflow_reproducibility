#!/bin/bash

#SBATCH -o h-otfm-combosciplex.out

#SBATCH -e h-otfm-combosciplex.err

#SBATCH -J h-otfm-combosciplex

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --mem=20G

#SBATCH -t 0-01:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python ../train_combosciplex.py --multirun dataset=combosciplex +hparams_search=hparams_combosciplex model=combosciplex training=combosciplex logger=zebrafish launcher=gpu_normal
