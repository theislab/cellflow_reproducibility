#!/bin/bash

# Read the JSON array from the file into a Bash array
#mapfile -t list_preds < <(jq -r '.[]' /lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_cytokine/all_preds.json)

declare -a list_preds=(
    wild-river-599_Donor9_OX40L_11_preds.h5ad
    peach-dust-595_Donor9_APRIL_11_preds.h5ad
    scarlet-cloud-591_Donor9_IFN-epsilon_11_preds.h5ad
    upbeat-forest-597_Donor9_Noggin_11_preds.h5ad
    stilted-wood-598_Donor9_OSM_11_preds.h5ad
    vibrant-frog-593_Donor9_4-1BBL_11_preds.h5ad
    flowing-forest-596_Donor9_BAFF_11_preds.h5ad
    exalted-wind-592_Donor9_C5a_11_preds.h5ad
    generous-frost-590_Donor9_IFN-beta_11_preds.h5ad
    breezy-rain-601_Donor9_IL-15_11_preds.h5ad
    worthy-firefly-594_Donor9_ADSF_11_preds.h5ad
)
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
python compute_metrics_new_cytokine.py "${split}"
EOF
done
