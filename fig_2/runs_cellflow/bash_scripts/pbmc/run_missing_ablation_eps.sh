#!/bin/bash

# Define the specific combinations as "index:epsilon_list"
# Extracted from the provided wandb_export CSV
declare -a tasks=(
    "0:0.01,0.1,1,10,100,1000000"
    "1:0.01,0.1,1,10,100,1000000"
    "2:0.01,0.1,1,10,100,1000000"
    "3:0.01,0.1,1,10,100,1000000"
    "4:0.01,0.1,1,10,100"
    "5:0.1,10"
    "6:0.01,10,1000000"
    "7:1000000"
    "10:1,1000000"
    "13:0.01,0.1,10"
    "18:0.01,1,10,100"
    "19:0.01,0.1,10,100,1000000"
    "20:0.01,0.1,1,10,100,1000000"
    "21:0.01,0.1,1,10,100,1000000"
    "22:0.01,0.1,100,1000000"
    "23:0.01,100,1000000"
)

# Loop through each task group
for task in "${tasks[@]}"; do
    # Split the string into the index and the comma-separated epsilon values
    IFS=":" read -r index eps_list <<< "$task"

    # Split the comma-separated epsilon_list into an array
    IFS="," read -r -a eps_array <<< "$eps_list"

    for eps in "${eps_array[@]}"; do
        # Submit an individual SLURM job per epsilon value
        sbatch <<EOF
#!/bin/bash
#SBATCH -o logs2/pbmc_idx${index}_eps${eps}_%j.out
#SBATCH -e logs2/pbmc_idx${index}_eps${eps}_%j.err
#SBATCH -J pbmc_${index}_${eps}
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


export WANDB_SERVICE_WAIT=300

# 2. Set a stable directory for wandb (replaces /tmp)
export WANDB_DIR=/home/icb/dominik.klein/tmp3
mkdir -p $WANDB_DIR

# Run Hydra with a single configuration (removed -m)
python /home/icb/dominik.klein/git_repos/ot_pert_new/fig_2/runs_cellflow/train_pbmc_new_donor.py \
    dataset=pbmc_new_donor \
    model=pbmc_new_donor \
    training=pbmc_new_donor_ablation_batch_size \
    logger=zebrafish \
    dataset.donor_held_out='Donor1' \
    dataset.idx_given_donor=$index \
    model.epsilon=$eps
EOF
        # Slight delay to avoid overwhelming the scheduler
        sleep 0.1
    done
done