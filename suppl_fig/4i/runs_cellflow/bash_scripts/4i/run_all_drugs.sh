#!/bin/bash

#SBATCH -o 3all_4i.out

#SBATCH -e 3all_4i.err

#SBATCH -J 3all_4i

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=160G

#SBATCH -t 0-24:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

python ../../train_4i.py --multirun dataset=4i logger=4i training=4i model=4i +hparams_search=run_all_drugs launcher=gpu_normal
