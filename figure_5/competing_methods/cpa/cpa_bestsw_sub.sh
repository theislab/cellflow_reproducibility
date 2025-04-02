#!/bin/bash
# BSUB -J cpa_bestsw
# BSUB -W 24:00
# BSUB -n 1
# BSUB -M 5G
# BSUB -q long
# BSUB -o /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/logs/cpa_ed.out
# BSUB -e /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/logs/cpa_ed.err

ml Miniforge3
conda activate py310_cf_stable

CONF_SW="/pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/configs/bnchmrk/sw_cpa.yml"
RUN_ID="20250110_191253_264"
COMBS="RA+BMP4,RA+CHIR+BMP4,CHIR+BMP4,XAV+BMP4,XAV+SHH,CHIR+SHH,RA+SHH,FGF8+BMP4,FGF8+SHH,RA+CHIR,FGF8+CHIR,RA+CHIR+SHH,FGF8+CHIR+BMP4,FGF8+CHIR+SHH"

python /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/cpa_bestsw.py $CONF_SW $RUN_ID $COMBS