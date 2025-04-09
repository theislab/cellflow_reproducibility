#!/bin/bash
#SBATCH -o launch.out
#SBATCH -e launch.err
#SBATCH -J launch
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH -t 0-00:10:00
#SBATCH --nice=1

DONORS=(Donor1 Donor2 Donor3 Donor4 Donor5 Donor6 Donor7 Donor8 Donor9 Donor10 Donor11 Donor12)

for DONOR in "${DONORS[@]}"; do
  sbatch --job-name=cyto_${DONOR} run_single_donor_cytokine_similarity.sh "$DONOR"
done
