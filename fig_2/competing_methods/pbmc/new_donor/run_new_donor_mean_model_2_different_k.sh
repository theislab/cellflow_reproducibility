#!/bin/bash

# Define an array of splits (your specified variables, one per line)
declare -a splits=(
    'Donor1'
    'Donor2'
    'Donor3'
    'Donor4'
    'Donor5'
    'Donor6'
    'Donor7'
    'Donor8'
    'Donor9'
    'Donor10'
    'Donor11'
    'Donor12'
)

# Loop through each split and each index
for split in "${splits[@]}"; do
    for index in {15,16,17}; do #{0..25}; do
        # Construct a SLURM job script and submit
        sbatch <<EOF
#!/bin/bash
#SBATCH -o mean2_"$split"_"$index".out
#SBATCH -e mean2_"$split"_"$index".err
#SBATCH -J mean2_"$split"_"$index"
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=300G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

# Print info
echo "Running on: \$(hostname)"
echo "Split: $split, Index: $index"

# Run your program with the split and index as arguments
python new_donor_mean_model_2_different_k.py "$split" "$index"
EOF
    done
done
