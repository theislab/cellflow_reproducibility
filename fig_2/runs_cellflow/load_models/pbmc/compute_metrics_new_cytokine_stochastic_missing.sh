#!/bin/bash

# Read the JSON array from the file into a Bash array
list_preds=("cosmic-haze-55_0.1_all_preds.h5ad")

# Loop through each split and submit a job
for split in "${list_preds[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o miss_metrics_${split}.out
#SBATCH -e miss_metrics_${split}.err
#SBATCH -J miss_metrics_${split}
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cellflow

# Print info
echo "Running on: \$(hostname)"
echo "Split: ${split}"

# Run your program with the split as an argument
python compute_metrics_new_cytokine_stochastic.py "${split}"
EOF
done
