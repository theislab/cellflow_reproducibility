#!/bin/bash

# Define an array of splits (your specified variables, one per line)
declare -a splits=(
    '4-1BBL'
    'ADSF'
    'APRIL'
    'BAFF'
    'C3a'
    'C5a'
    'CD27L'
    'CD30L'
    'CD40L'
    'CT-1'
    'Decorin'
    'EGF'
    'EPO'
    'FGF-beta'
    'FLT3L'
    'FasL'
    'G-CSF'
    'GDNF'
    'GITRL'
    'GM-CSF'
    'HGF'
    'IFN-alpha1'
    'IFN-beta'
    'IFN-epsilon'
    'IFN-gamma'
    'IFN-lambda1'
    'IFN-lambda2'
    'IFN-lambda3'
    'IFN-omega'
    'IGF-1'
    'IL-1-alpha'
    'IL-1-beta'
    'IL-10'
    'IL-11'
    'IL-12'
    'IL-13'
    'IL-15'
    'IL-16'
    'IL-17A'
    'IL-17B'
    'IL-17C'
    'IL-17D'
    'IL-17E'
    'IL-17F'
    'IL-18'
    'IL-19'
    'IL-1Ra'
    'IL-2'
    'IL-20'
    'IL-21'
    'IL-22'
    'IL-23'
    'IL-24'
    'IL-26'
    'IL-27'
    'IL-3'
    'IL-31'
    'IL-32-beta'
    'IL-33'
    'IL-34'
    'IL-35'
    'IL-36-alpha'
    'IL-36Ra'
    'IL-4'
    'IL-5'
    'IL-6'
    'IL-7'
    'IL-8'
    'IL-9'
    'LIF'
    'LIGHT'
    'LT-alpha1-beta2'
    'LT-alpha2-beta1'
    'Leptin'
    'M-CSF'
    'Noggin'
    'OSM'
    'OX40L'
    'PBS'
    'PRL'
    'PSPN'
    'RANKL'
    'SCF'
    'TGF-beta1'
    'TL1A'
    'TNF-alpha'
    'TPO'
    'TRAIL'
    'TSLP'
    'TWEAK'
    'VEGF'
)

# Loop through each split and submit a job
for split in "${splits[@]}"; do
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o make_split_"$split".out
#SBATCH -e make_split_"$split".err
#SBATCH -J make_split_"$split"
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=300G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp

# Print info
echo "Running on: \$(hostname)"
echo "Split: $split"

# Run your program with the split as an argument
python make_splits_new_cytokine.py "$split"
EOF
done
