#!/bin/bash
# BSUB -J PSWEEP_RAY
# BSUB -W 72:00
# BSUB -n 4
# BSUB -M 32G
# BSUB -q long
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -R "span[ptile=1]" 
# BSUB -o logs/PSWEEP_RAY_%J.out


ml Miniforge3
conda activate $RAY_CONDA_ENV

bash /home/fleckj/projects/cellflow/scripts/ray/ray_launch_cluster.sh -c "python $RAY_JOB_SCRIPT $RAY_JOB_CONFIG" -n $RAY_CONDA_ENV
