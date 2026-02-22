#!/bin/bash

# Read the JSON array from the file into a Bash array
mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_cytokine_stochastic_2/pred_files_round_2.json)

# Loop through each split and submit a job
for split in "${list_preds[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o metrics_${split}.out
#SBATCH -e metrics_${split}.err
#SBATCH -J metrics_${split}
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
