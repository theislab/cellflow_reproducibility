#!/bin/bash
#SBATCH --job-name=SCORE_CL
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=60Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_short
#SBATCH --nice=0
##SBATCH -w gpusrv64

source /home/icb/alejandro.tejada/.bashrc
conda activate flow_matching
python -u /home/icb/alejandro.tejada/ot_pert_reproducibility/jaxtry.py
