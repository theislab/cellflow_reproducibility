#!/bin/bash

# Define an array of tuples (split, model_name)
declare -a tuples=(
    "concatenated_drug_embeddings"
    "direct_embedding"
    "closest_cell_line"
)

# Loop through each tuple and submit a job
for tuple in "${tuples[@]}"; do
    # Read split and model_name from the tuple
    read -r emb_type <<< "$tuple"
    
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o eval_sciplex_"$emb_type".out
#SBATCH -e eval_sciplex_"$emb_type".err
#SBATCH -J eval_sciplex_"$emb_type"
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=140G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow


# Print info
echo "Running on: \$(hostname)"
echo "Split: $emb_type"

# Run your program with the tuple as an argument
python evaluate_sciplex.py "$emb_type"
EOF
done
