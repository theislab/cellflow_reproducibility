#!/bin/bash

# Define an array of splits (your specified variables, one per line)
declare -a splits=(
    '4-1BBL'
    'ADSF'
    'APRIL'
    'BAFF'
    'C5a'
    'IFN-beta'
    'IL-13'
    'IL-15'
    'Noggin'
    'OSM'
    'OX40L'
    'IFN-epsilon'
)

# Loop through each split and each index
for split in "${splits[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -o id_"$split".out
#SBATCH -e id_"$split".err
#SBATCH -J id_"$split"
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Print info
echo "Running on: \$(hostname)"
echo "Split: $split"

# Run your program with the split and index as arguments
python new_cytokine_identity.py "$split"
EOF
done
