#!/bin/bash

# first batch
#mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor/all_preds.json)

# now for completely unseen donors
mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor/preds_0_seen.json)

# Loop through each split and submit a job
for split in "${list_preds[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o d_metrics_${split}.out
#SBATCH -e d_metrics_${split}.err
#SBATCH -J d_metrics_${split}
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

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
done
