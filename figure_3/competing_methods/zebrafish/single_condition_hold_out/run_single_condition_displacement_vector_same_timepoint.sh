#!/bin/bash

#SBATCH -o same_t_zebra.out

#SBATCH -e same_t_zebra.err

#SBATCH -J same_t_zebra

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=2

#SBATCH --constraint=a100_80gb|h100_80gb

#SBATCH --mem-per-cpu=120G

#SBATCH -t 3-00:00:00

#SBATCH --nice=1

source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python single_condition_displacement_vector_same_timepoint.py
