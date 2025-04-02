#!/bin/bash
# BSUB -J MAP_SPLIT
# BSUB -W 10:00
# BSUB -n 1
# BSUB -M 64G
# BSUB -q long
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -o logs/MAP_SPLIT_%J.out


ml Miniforge3
conda activate py310_atlas

echo $SPLIT_DIR

python /home/fleckj/projects/cellflow/scripts/data/preproc_and_map_split.py $SPLIT_DIR
