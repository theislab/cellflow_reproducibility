#!/bin/bash

#SBATCH -o gears_one_satija.out

#SBATCH -e gears_one_satija.err

#SBATCH -J gears_one_satija

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --constraint=a100_40gb|a100_80gb

#SBATCH --gres=gpu:1

#SBATCH --mem=90G

#SBATCH -t 00:10:00

#SBATCH --nice=0

source /home/icb/alejandro.tejada/.bashrc
conda activate gears

python /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_gears/train_norman_2seen.py