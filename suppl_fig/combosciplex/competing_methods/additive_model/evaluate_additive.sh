#!/bin/bash

# Define an array of tuples (split, model_name)
declare -a tuples=(
    "1 "
    "2 "
    "3 "
    "4 "
)

# Loop through each tuple and submit a job
for tuple in "${tuples[@]}"; do
    # Read split and model_name from the tuple
    read -r split model_name <<< "$tuple"
    
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o eval_add_"$split".out
#SBATCH -e eval_add_"$split".err
#SBATCH -J eval_add_"$split"
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp


# Print info
echo "Running on: \$(hostname)"
echo "Split: $split"

# Run your program with the tuple as an argument
python evaluate_additive.py "$split" 
EOF
done
