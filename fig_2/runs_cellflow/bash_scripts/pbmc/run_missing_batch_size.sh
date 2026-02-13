#!/bin/bash

# Define the specific combinations as "index:tau_list"
declare -a tasks=(
    "0:1.0,0.99,0.95,0.8"
    "1:1.0,0.99,0.95,0.9,0.8"
    "2:0.99,0.95,0.9,0.8"
    "3:1.0,0.99,0.95,0.9,0.8"
    "4:0.95,0.9,0.8"
    "5:1.0,0.8"
    "7:0.99"
    "9:0.95,0.8"
    "10:1.0"
    "16:0.99"
    "17:0.99,0.95"
    "19:0.99,0.9"
    "21:0.99,0.95"
    "22:0.99,0.95"
)

# Loop through each task group
for task in "${tasks[@]}"; do
    # Split the string into the index and the comma-separated tau values
    IFS=":" read -r index tau_list <<< "$task"

    # Split the comma-separated tau_list into an array to iterate over
    IFS="," read -r -a tau_array <<< "$tau_list"

    for tau in "${tau_array[@]}"; do
        # Construct and submit an individual SLURM job per tau value
        # Note: -m is removed from the python command
        sbatch <<EOF
#!/bin/bash
#SBATCH -o logs/pbmc_idx${index}_tau${tau}_%j.out
#SBATCH -e logs/pbmc_idx${index}_tau${tau}_%j.err
#SBATCH -J pbmc_${index}_${tau}
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

# Run Hydra for a SINGLE configuration
python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py \
    dataset=pbmc_new_donor \
    model=pbmc_new_donor \
    training=pbmc_new_donor_ablation_batch_size \
    logger=zebrafish \
    dataset.donor_held_out='Donor1' \
    dataset.idx_given_donor=$index \
    model.tau_a=$tau
EOF
        # Brief sleep to avoid hitting the scheduler too fast
        sleep 0.1
    done
done