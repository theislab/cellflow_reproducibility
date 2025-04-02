#!/bin/bash
# e.g. ./launch_ray_job.sh sweep_script.py sweep_config.yaml py310_dev

RAY_SUB_SCRIPT="/home/fleckj/projects/cellflow/scripts/ray/allocate_resources.sh"

# Submit Ray job with modified env vars
USE_ENV="all, RAY_JOB_SCRIPT=$1, RAY_JOB_CONFIG=$2, RAY_CONDA_ENV=$3"
bsub -env "$USE_ENV" < $RAY_SUB_SCRIPT

