#!/bin/bash
#SBATCH -o .train_slurm_hp_1seen.out
#SBATCH -e .train_slurm_hp_1seen.err
#SBATCH -J train_slurm_hp
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH -c 4
#SBATCH --mem=90G
#SBATCH -t 2-00:00

source $HOME/.bashrc
mamba activate gears

python -u ../train_norman.py dataset=norman_1seen

