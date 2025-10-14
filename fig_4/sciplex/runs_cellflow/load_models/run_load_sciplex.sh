#!/bin/bash

# Define an array of tuples (split, model_name)
declare -a tuples=(
    "10 dainty-lake-1 closest_cell_line"
    "10 fiery-brook-1 concatenated_drug_embeddings"
    "10 prime-pyramid-3 direct_embedding"
    )

# Loop through each tuple and submit a job
for tuple in "${tuples[@]}"; do
    # Read split and model_name from the tuple
    read -r split model_name emb_type <<< "$tuple"
    
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o load_sciplex_"$model_name".out
#SBATCH -e load_sciplex_"$model_name".err
#SBATCH -J load_sciplex_"$model_name"
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-24:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow


# Print info
echo "Running on: \$(hostname)"
echo "Split: $split, Model: $model_name"

# Run your program with the tuple as an argument
python load_sciplex.py "$split" "$model_name" "$emb_type"
EOF
done
