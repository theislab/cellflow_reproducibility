#!/bin/bash

# Read the JSON array from the file into a Bash array
mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor/all_preds.json)

# Select the first item (or any specific item) from the array
split=${list_preds[0]} # Change the index to choose a different item if needed

# Construct a SLURM job script and submit
sbatch <<EOF
#!/bin/bash
#SBATCH -o 2d_metrics_${split}.out
#SBATCH -e 2d_metrics_${split}.err
#SBATCH -J 2d_metrics_${split}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=300G
#SBATCH -t 0-12:00:00
#SBATCH --nice=0

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Print info
echo "Running on: \$(hostname)"
echo "Split: ${split}"

# Run your program with the split as an argument
python compute_metrics_new_donor.py "${split}"
EOF
