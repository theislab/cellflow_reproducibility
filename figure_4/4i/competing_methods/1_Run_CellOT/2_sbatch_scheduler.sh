#!/bin/bash
#
#SBATCH --job-name=CellOT_job
#SBATCH --output=/home/icb/manuel.gander/do_k.txt
#SBATCH --error=/home/icb/manuel.gander/error_k.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=20G
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

conda init bash
conda activate cellot

#cd pert
#cd notebooks
#cd Cellot

k="$1"
python /home/icb/manuel.gander/pert/cellot-main/scripts/train.py --outdir /home/icb/manuel.gander/pert/cellot-main/results/4i/drug-$k/model-cellot --config /home/icb/manuel.gander/pert/cellot-main/configs/tasks/4i.yaml --config /home/icb/manuel.gander/pert/cellot-main/configs/models/cellot.yaml --config.data.target $k