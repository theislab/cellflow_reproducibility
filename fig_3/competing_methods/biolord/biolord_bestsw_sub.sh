#!/bin/bash
# BSUB -J biolord_bestsw
# BSUB -W 24:00
# BSUB -n 1
# BSUB -M 5G
# BSUB -q long
# BSUB -o /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/logs/biolord_bestsw.out
# BSUB -e /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/logs/biolord_bestsw.err

ml Miniforge3
conda activate py310_cf_biolord

CONF_SW="/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/configs/bnchmrk/sw_biolord.yml"
RUN_ID="20241209_034509_262"
COMBS="RA+BMP4,RA+CHIR+BMP4,CHIR+BMP4,XAV+BMP4,XAV+SHH,CHIR+SHH,RA+SHH,FGF8+BMP4,FGF8+SHH,RA+CHIR,FGF8+CHIR,RA+CHIR+SHH,FGF8+CHIR+BMP4,FGF8+CHIR+SHH"

python /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/biolord_bestsw.py $CONF_SW $RUN_ID $COMBS