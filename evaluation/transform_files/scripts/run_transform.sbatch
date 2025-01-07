#!/bin/bash

#SBATCH -o ./logs/transform_3.out

#SBATCH -e ./logs/transform_3.err

#SBATCH -J transform_3

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=140G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

source ${HOME}/.bashrc
conda activate ot_pert

cd /home/icb/alessandro.palma/environment/ot_pert/ot_pert_reproducibility/evaluation/transform_files
 
# python save_transformed_files.py 0 
# python save_transformed_files.py 1
# python save_transformed_files.py 2
python save_transformed_files.py 3