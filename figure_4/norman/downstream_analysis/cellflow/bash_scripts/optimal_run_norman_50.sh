#!/bin/bash

#SBATCH -o h-otfm-norman_%j.out

#SBATCH -e h-otfm-norman_%j.err

#SBATCH -J h-otfm-norman

#SBATCH -p gpu_p

#SBATCH --qos=gpu_long

#SBATCH --gres=gpu:1

#SBATCH --mem=80G

#SBATCH -t 0-12:00:00

#SBATCH --nice=1

SPLIT="${1}"
echo "split: ${SPLIT}"

source /home/haicu/soeren.becker/.bashrc
conda activate /home/haicu/soeren.becker/miniconda3/envs/env_cfp2

python /home/haicu/soeren.becker/repos/ot_pert_reproducibility/runs_otfm/train_norman_pca_50.py \
    dataset=norman \
    dataset.split="${SPLIT}" \
    +hparams_search=hparams_norman_select_0_tough_leaf_242 \
    model=norman \
    training=norman \
    ++training.save_model=true \
    ++training.save_predictions=true \
    logger=norman
    
