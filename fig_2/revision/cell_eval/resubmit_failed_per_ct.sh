#!/bin/bash
# Resubmit only the failed per-ct jobs (57 jobs that hit "Illegal instruction" due to polars CPU check).
# Adds POLARS_SKIP_CPU_CHECK=1 to bypass the issue.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/icb/dominik.klein/mambaforge/envs/cell_eval/bin/python"
SCRIPT="${SCRIPT_DIR}/compute_cell_eval_metrics_per_ct.py"
OUTDIR="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_cell_eval_per_ct"
LOG_DIR="${SCRIPT_DIR}/logs_per_ct"

# --- Failed mean_model_1 jobs ---
declare -A FAILED_MM1
FAILED_MM1[Donor8]="17"
FAILED_MM1[Donor9]="15 16 17"
FAILED_MM1[Donor10]="15 16 17"
FAILED_MM1[Donor11]="15 16 17"
FAILED_MM1[Donor12]="16 17"

for donor in "${!FAILED_MM1[@]}"; do
    for split_idx in ${FAILED_MM1[$donor]}; do
        sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_mean_model_1_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_mean_model_1_${donor}_${split_idx}.err
#SBATCH -J cect_mm1_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

export POLARS_SKIP_CPU_CHECK=1

${PYTHON} ${SCRIPT} \
    --method mean_model_1 \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
    done
done

# --- Failed mean_model_2 jobs (all 12 donors, various splits) ---
declare -A FAILED_MM2
FAILED_MM2[Donor1]="15 16 17"
FAILED_MM2[Donor2]="15 16 17"
FAILED_MM2[Donor3]="15 16 17"
FAILED_MM2[Donor4]="15 16 17"
FAILED_MM2[Donor5]="15 16 17"
FAILED_MM2[Donor6]="15 16 17"
FAILED_MM2[Donor7]="15 16 17"
FAILED_MM2[Donor8]="15 16 17"
FAILED_MM2[Donor9]="15"
FAILED_MM2[Donor11]="16 17"
FAILED_MM2[Donor12]="15 16 17"

for donor in "${!FAILED_MM2[@]}"; do
    for split_idx in ${FAILED_MM2[$donor]}; do
        sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_mean_model_2_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_mean_model_2_${donor}_${split_idx}.err
#SBATCH -J cect_mm2_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

export POLARS_SKIP_CPU_CHECK=1

${PYTHON} ${SCRIPT} \
    --method mean_model_2 \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
    done
done

# --- Failed closest_embedding jobs ---
declare -A FAILED_CE
FAILED_CE[Donor1]="15 16 17"
FAILED_CE[Donor2]="15 16 17"
FAILED_CE[Donor5]="15 16 17"
FAILED_CE[Donor6]="15 16 17"
FAILED_CE[Donor7]="15 17"

for donor in "${!FAILED_CE[@]}"; do
    for split_idx in ${FAILED_CE[$donor]}; do
        sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_closest_embedding_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_closest_embedding_${donor}_${split_idx}.err
#SBATCH -J cect_ce_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

export POLARS_SKIP_CPU_CHECK=1

${PYTHON} ${SCRIPT} \
    --method closest_embedding \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
    done
done

# --- Failed cellflow job ---
sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_cellflow_Donor3_earnest-plant-93.out
#SBATCH -e ${LOG_DIR}/ce_cellflow_Donor3_earnest-plant-93.err
#SBATCH -J cect_cf_Donor3_earnest-plant-93
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

export POLARS_SKIP_CPU_CHECK=1

${PYTHON} ${SCRIPT} \
    --method cellflow \
    --donor Donor3 \
    --wandb-name earnest-plant-93 \
    --num-train-cytos 65 \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF

echo "Resubmitted: 12 (mean_model_1) + 30 (mean_model_2) + 14 (closest_embedding) + 1 (cellflow) = 57 jobs"
