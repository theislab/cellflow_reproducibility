#!/bin/bash

# Define an array of splits (your specified variables, one per line)
declare -a splits=(
    "Donor1"
    "Donor2"
    "Donor3"
    "Donor4"
    "Donor5"
    "Donor6"
    "Donor7"
    "Donor8"
    "Donor9"
    "Donor10"
    "Donor11"
    "Donor12"
)

# Loop through each split and submit a job
for split in "${splits[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o make_split_"$split".out
#SBATCH -e make_split_"$split".err
#SBATCH -J make_split_"$split"
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=300G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Print info
echo "Running on: \$(hostname)"
echo "Split: $split"

# Run your program with the split as an argument
python add_split_new_donor_zero_shot.py "$split"
EOF
done
