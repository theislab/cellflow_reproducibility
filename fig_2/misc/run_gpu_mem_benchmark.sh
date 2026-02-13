#!/bin/bash
set -euo pipefail

for batch_size in 64 128 256 512 1024 2048 4096 8192 16384; do
  sbatch <<EOF
#!/bin/bash
#SBATCH -o gpu_mem_${batch_size}_%j.out
#SBATCH -e gpu_mem_${batch_size}_%j.err
#SBATCH -J gpu_mem_${batch_size}
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --constraint=a100_80gb
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

set -euo pipefail

source "/home/icb/dominik.klein/mambaforge/etc/profile.d/conda.sh"
conda activate "/home/icb/dominik.klein/mambaforge/envs/cellflow"

cd "\$SLURM_SUBMIT_DIR"
python "\$SLURM_SUBMIT_DIR/gpu_mem_benchmark.py" "${batch_size}"
EOF
done
