#!/bin/bash

mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor_ablation_batch_size_sa276.json)

# Loop through each split and submit a job
for split in "${list_preds[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o d_bs_${split}.out
#SBATCH -e d_bs_${split}.err
#SBATCH -J d_bs_${split}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
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
python compute_metrics_new_donor_ablation.py "${split}" "batch_size"
EOF
done
