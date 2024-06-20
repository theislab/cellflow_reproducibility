#!/bin/bash
#SBATCH -o ./logs/train_slurm_hp2.out
#SBATCH -e ./logs/train_slurm_hp2.err
#SBATCH -J train_slurm_hp
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -c 4
#SBATCH --mem=90G
#SBATCH -t 2-00:00

source $HOME/.bashrc
conda activate ot_pert_genot

python ../train.py --multirun launcher=slurm_icb logger=hparam_search 

