#!/bin/bash

#SBATCH -o h-genot-norman-%j.out
#SBATCH -e h-genot-norman-%j.err
#SBATCH -J h-genot-norman
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=90G
#SBATCH -t 48:00:00
#SBATCH --nice=1

source ${HOME}/.bashrc
mamba activate jax

python -u /home/icb/alejandro.tejada/ot_pert_reproducibility/runs_genot/train_norman.py --multirun dataset=norman_2seen hparams_search=hparams_norman model=norman training=training_norman logger=norman_2seen
