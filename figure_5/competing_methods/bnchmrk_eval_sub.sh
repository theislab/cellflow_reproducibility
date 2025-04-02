#!/bin/bash

CONFIG_PATH=$1
shift
BSUB_PARAMS=$@

# Create a temporary script for the BSUB job
TEMP_SCRIPT=$(mktemp)

cat <<EOT > $TEMP_SCRIPT
#!/bin/bash
#BSUB $BSUB_PARAMS

eval "\$(conda shell.bash hook)"
conda activate py310_cf_stable

python /pmount/projects/site/pred/organoid-atlas/USERS/bobrovsd/scripts/bnchmrk/bnchmrk_eval.py $CONFIG_PATH
EOT

# Submit the BSUB job
bsub < $TEMP_SCRIPT

# Clean up the temporary script
rm $TEMP_SCRIPT