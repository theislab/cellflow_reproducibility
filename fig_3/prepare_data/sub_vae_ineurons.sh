#!/bin/bash
# BSUB -J vae_ineurons_single
# BSUB -W 24:00
# BSUB -n 1
# BSUB -M 100G
# BSUB -q long
# BSUB -R "span[hosts=1]"
# BSUB -gpu "num=1:j_exclusive=yes"
# BSUB -o /home/bobrovsd/logs/vae_ineurons_single.out
# BSUB -e /home/bobrovsd/logs/vae_ineurons_single.err

python /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/embed/vae_ineurons.py