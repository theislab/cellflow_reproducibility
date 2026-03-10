#!/bin/bash
# Resubmit only the missing global metric jobs:
# - 20 mean_model_1 runs
# - 9 closest_embedding runs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/icb/dominik.klein/mambaforge/envs/cell_eval/bin/python"
SCRIPT="${SCRIPT_DIR}/compute_cell_eval_metrics.py"
OUTDIR="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_cell_eval"
LOG_DIR="${SCRIPT_DIR}/logs"

# --- Missing mean_model_1 (20 jobs) ---
MISSING_MM1="Donor2_15 Donor3_17 Donor4_15 Donor4_16 Donor6_16 Donor6_17 Donor7_15 Donor7_16 Donor8_15 Donor8_17 Donor9_15 Donor9_16 Donor9_17 Donor10_15 Donor10_16 Donor10_17 Donor11_15 Donor11_16 Donor11_17 Donor12_15"

for entry in $MISSING_MM1; do
    donor="${entry%_*}"
    split_idx="${entry##*_}"
    sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_mean_model_1_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_mean_model_1_${donor}_${split_idx}.err
#SBATCH -J ce_mean_model_1_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

${PYTHON} ${SCRIPT} \
    --method mean_model_1 \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
done

# --- Missing closest_embedding (9 jobs) ---
MISSING_CE="Donor1_15 Donor2_17 Donor4_17 Donor5_16 Donor5_17 Donor6_15 Donor6_17 Donor7_15 Donor7_16"

for entry in $MISSING_CE; do
    donor="${entry%_*}"
    split_idx="${entry##*_}"
    sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_closest_embedding_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_closest_embedding_${donor}_${split_idx}.err
#SBATCH -J ce_closest_embedding_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

${PYTHON} ${SCRIPT} \
    --method closest_embedding \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
done

echo "Resubmitted: 20 (mean_model_1) + 9 (closest_embedding) = 29 jobs"
