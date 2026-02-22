#!/bin/bash

# Define the specific combinations as "index:batch_size_list"
declare -a tasks=(
    "0:1024"
    "1:1024,2048"
)

# Loop through each task group
for task in "${tasks[@]}"; do
    # Split the string into the index and the comma-separated batch size values
    IFS=":" read -r index bs_list <<< "$task"

    # Split the comma-separated bs_list into an array to iterate over
    IFS="," read -r -a bs_array <<< "$bs_list"

    for bs in "${bs_array[@]}"; do
        # Construct and submit an individual SLURM job per batch size value
        sbatch <<EOF
#!/bin/bash
#SBATCH -o logs_bs_missing/pbmc_idx${index}_bs${bs}_%j.out
#SBATCH -e logs_bs_missing/pbmc_idx${index}_bs${bs}_%j.err
#SBATCH -J pbmc_${index}_${bs}
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb"
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH -t 1-00:00:00
#SBATCH --nice=1

# Environment setup
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Use a job-local temp directory to prevent wandb port file errors
export TMPDIR=\${TMPDIR:-/tmp}/\${SLURM_JOB_ID}
mkdir -p \$TMPDIR

# Run Hydra for a SINGLE configuration
python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py \
    dataset=pbmc_new_donor \
    model=pbmc_new_donor \
    training=pbmc_new_donor_ablation_batch_size \
    logger=zebrafish \
    dataset.donor_held_out='Donor1' \
    dataset.idx_given_donor=$index \
    training.batch_size=$bs
EOF
        # Brief sleep to avoid hitting the scheduler too fast
        sleep 0.1
    done
done
