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
    for index in {0..42}; do
        # Construct a SLURM job script and submit
        sbatch <<EOF
#!/bin/bash
#SBATCH -o mean2_"$split"_"$index".out
#SBATCH -e mean2_"$split"_"$index".err
#SBATCH -J mean2_"$split"_"$index"
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
echo "Split: $split, Index: $index"

# Run your program with the split and index as arguments
python new_cytokine_mean_model_2.py "$split" "$index"
EOF
    done
done
