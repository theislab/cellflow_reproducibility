#!/bin/bash

# Define the specific combinations as "index:tau_list"
declare -a tasks=(
    "0:0.9"
    "2:1.0"
    "3:0.95"
    "4:0.99,1.0"
    "5:0.9,0.95,0.99"
    "6:0.8,0.9,0.95,0.99,1.0"
    "7:0.8,0.9,0.95,1.0"
    "8:0.8,0.9,0.95,0.99,1.0"
    "9:0.9,0.99,1.0"
    "10:0.8,0.9,0.95,0.99"
    "11:0.8,0.9,0.95,0.99,1.0"
    "12:0.8,0.9,0.95,0.99,1.0"
    "13:0.8,0.9,0.95,0.99,1.0"
    "14:0.8,0.9,0.95,0.99,1.0"
    "15:0.8,0.9,0.95,0.99,1.0"
    "16:0.8,0.9,0.95,1.0"
    "17:0.8,0.9,1.0"
    "18:0.8,0.9,0.95,0.99,1.0"
    "19:0.8,0.95,1.0"
    "20:0.8,0.9,0.95,0.99,1.0"
    "21:0.8,0.9,0.95,0.99,1.0"
    "22:0.8,0.9,1.0"
    "23:0.8,0.9,0.95,0.99,1.0"
)

# Loop through each task group
for task in "${tasks[@]}"; do
    # Split the string into the index and the comma-separated tau values
    IFS=":" read -r index tau_list <<< "$task"

    # Split the comma-separated tau_list into an array to iterate over
    IFS="," read -r -a tau_array <<< "$tau_list"

    for tau in "${tau_array[@]}"; do
        # Construct and submit an individual SLURM job per tau value
        sbatch <<EOF
#!/bin/bash
#SBATCH -o logs_u/pbmc_idx${index}_tau${tau}_%j.out
#SBATCH -e logs_u/pbmc_idx${index}_tau${tau}_%j.err
#SBATCH -J pbmc_${index}_${tau}
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb"
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH -t 1-00:00:00
#SBATCH --nice=1

# Use a job-local temp directory to prevent wandb port file errors
export TMPDIR=\$HOME/tmp/\${SLURM_JOB_ID}
mkdir -p \$TMPDIR

# Environment setup
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Run Hydra for a SINGLE configuration
python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py \
    dataset=pbmc_new_donor \
    model=pbmc_new_donor \
    training=pbmc_new_donor_ablation_unbalancedness \
    logger=zebrafish \
    dataset.donor_held_out='Donor1' \
    dataset.idx_given_donor=$index \
    model.tau_a=$tau
EOF
        # Brief sleep to avoid hitting the scheduler too fast
        sleep 0.1
    done
done
