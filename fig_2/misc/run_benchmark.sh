#!/bin/bash
set -euo pipefail

# --- Configuration ---
OUT_DIR="/lustre/groups/ml01/workspace/ot_perturbation/results/profiling"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- A100 80GB (high-end) ---
sbatch <<EOF
#!/bin/bash
#SBATCH -o ${OUT_DIR}/benchmark_a100_%j.out
#SBATCH -e ${OUT_DIR}/benchmark_a100_%j.err
#SBATCH -J benchmark_a100
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --constraint=a100_80gb
#SBATCH -t 0-06:00:00
#SBATCH --nice=1

set -euo pipefail

source "/home/icb/dominik.klein/mambaforge/etc/profile.d/conda.sh"
conda activate "/home/icb/dominik.klein/mambaforge/envs/cellflow"

cd "${SCRIPT_DIR}"

python benchmark.py \
    --batch_size 1024 \
    --train_iters 100 \
    --n_cells 1000 5000 10000 50000 \
    --out_dir "${OUT_DIR}" \
    --out_prefix benchmark_a100
EOF

echo "Submitted A100 80GB benchmark job"

# --- V100 32GB (consumer-comparable) ---
sbatch <<EOF
#!/bin/bash
#SBATCH -o ${OUT_DIR}/benchmark_v100_%j.out
#SBATCH -e ${OUT_DIR}/benchmark_v100_%j.err
#SBATCH -J benchmark_v100
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --constraint=v100_32gb
#SBATCH -t 0-06:00:00
#SBATCH --nice=1

set -euo pipefail

source "/home/icb/dominik.klein/mambaforge/etc/profile.d/conda.sh"
conda activate "/home/icb/dominik.klein/mambaforge/envs/cellflow"

cd "${SCRIPT_DIR}"

python benchmark.py \
    --batch_size 1024 \
    --train_iters 100 \
    --n_cells 1000 5000 10000 50000 \
    --out_dir "${OUT_DIR}" \
    --out_prefix benchmark_v100
EOF

echo "Submitted V100 32GB benchmark job"
