#!/bin/bash
# Resubmit only the failed methods (mean_model_1, mean_model_2, cellflow).
# Identity and closest_embedding already succeeded.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/icb/dominik.klein/mambaforge/envs/cell_eval/bin/python"
SCRIPT="${SCRIPT_DIR}/compute_cell_eval_metrics.py"
OUTDIR="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_cell_eval"
LOG_DIR="${SCRIPT_DIR}/logs"

DONORS="Donor1 Donor2 Donor3 Donor4 Donor5 Donor6 Donor7 Donor8 Donor9 Donor10 Donor11 Donor12"
SPLIT_INDICES="15 16 17"

# --- mean_model_1, mean_model_2 ---
for method in mean_model_1 mean_model_2; do
    for donor in $DONORS; do
        for split_idx in $SPLIT_INDICES; do
            sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_${method}_${donor}_${split_idx}.out
#SBATCH -e ${LOG_DIR}/ce_${method}_${donor}_${split_idx}.err
#SBATCH -J ce_${method}_${donor}_${split_idx}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

${PYTHON} ${SCRIPT} \
    --method ${method} \
    --donor ${donor} \
    --split-idx ${split_idx} \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
        done
    done
done

# --- CellFlow ---
declare -A WANDB_NAMES
WANDB_NAMES[Donor1]="charmed-dragon-45 earthy-blaze-46 polar-cherry-43"
WANDB_NAMES[Donor2]="northern-serenity-71 olive-oath-68 scarlet-sun-67"
WANDB_NAMES[Donor3]="earnest-plant-93 electric-river-91 laced-feather-92"
WANDB_NAMES[Donor4]="denim-wood-115 fallen-plasma-117 whole-music-116"
WANDB_NAMES[Donor5]="fiery-water-140 neat-glitter-139 restful-darkness-141"
WANDB_NAMES[Donor6]="absurd-puddle-164 hopeful-bird-163 lilac-blaze-165"
WANDB_NAMES[Donor7]="elated-snowball-188 light-waterfall-189 radiant-frog-187"
WANDB_NAMES[Donor8]="classic-firefly-213 sage-sponge-212 true-shape-211"
WANDB_NAMES[Donor9]="blooming-sky-236 expert-gorge-235 laced-totem-237"
WANDB_NAMES[Donor10]="denim-jazz-261 firm-dew-259 swept-eon-260"
WANDB_NAMES[Donor11]="azure-bee-283 grateful-lion-285 spring-music-284"
WANDB_NAMES[Donor12]="flowing-sun-307 noble-star-308 winter-terrain-310"

for donor in $DONORS; do
    for wandb_name in ${WANDB_NAMES[$donor]}; do
        sbatch <<EOF
#!/bin/bash
#SBATCH -o ${LOG_DIR}/ce_cellflow_${donor}_${wandb_name}.out
#SBATCH -e ${LOG_DIR}/ce_cellflow_${donor}_${wandb_name}.err
#SBATCH -J ce_cf_${donor}_${wandb_name}
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=200G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

${PYTHON} ${SCRIPT} \
    --method cellflow \
    --donor ${donor} \
    --wandb-name ${wandb_name} \
    --num-train-cytos 65 \
    --outdir ${OUTDIR} \
    --num-threads 8
EOF
    done
done

echo "Resubmitted: 72 (mean_model) + 36 (cellflow) = 108 jobs"
