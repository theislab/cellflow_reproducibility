#!/bin/bash

#SBATCH -o otfm_one_satija_ifng_check.out

#SBATCH -e otfm_one_satija_ifng_check.err

#SBATCH -J otfm_one_satija_ifng_check

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH --mem=100G

#SBATCH -t 1-00:00

#SBATCH --nice=1

#source ${HOME}/.bashrc
source /lustre/groups/ml01/workspace/lea.zimmermann/software/miniconda3/etc/profile.d/conda.sh
conda activate cfp
#conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

python /home/icb/lea.zimmermann/projects/cell_flow_perturbation/run_otfm_sweep/train_satija_vae.py dataset=satija_ifng logger=satija_ifng training=satija_ifng launcher=slurm_icb model=satija_ifng
