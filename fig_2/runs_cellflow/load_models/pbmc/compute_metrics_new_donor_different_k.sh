#!/bin/bash

mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor/pred_files_64.json)


# Loop through each split and submit a job
for file in "${list_preds[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o diff_k/d_metrics_${file}.out
#SBATCH -e diff_k/d_metrics_${file}.err
#SBATCH -J diff_k/d_metrics_${file}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=400G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Print info
echo "Running on: \$(hostname)"
echo "Split: ${file}"

# Run your program with the split as an argument
python compute_metrics_new_donor_different_k.py "${file}"
EOF
done
