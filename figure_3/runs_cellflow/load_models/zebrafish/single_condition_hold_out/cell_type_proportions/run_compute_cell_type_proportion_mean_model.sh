#!/bin/bash

# Define an array of tuples (ood_cond, model_name)
declare -a tuples=(
    "hoxb1a_control_24 dazzling-shadow-71"
    "hoxb1a_control_36 vague-river-70"
    "tbxta_control_18 rural-pine-69"
    "epha4a_control_18 leafy-field-68"
    "tbx16_msgn1_18 rare-breeze-67"
    "tbxta_control_24 cool-capybara-66"
    "tbx16_control_24 charmed-sun-65"
    "tbx16_tbx16l_36 dashing-galaxy-64"
    "tbx16_control_36 dandy-planet-63"
    "cdx4_cdx1a_18 hopeful-mountain-62"
    "epha4a_control_36 fast-shape-61"
    "tbx1_control_24 genial-deluge-60"
    "foxd3_control_48 cosmic-blaze-58"
    "met_control_48 ancient-water-59"
    "cdx4_control_24 graceful-meadow-57"
    "tbxta_control_36 desert-salad-56"
    "hgfa_control_36 swift-wave-55"
    "tbx16_tbx16l_18 youthful-snowflake-54"
    "cdx4_cdx1a_24 ruby-tree-53"
    "hand2_control_36 solar-energy-51"
    "tfap2a_foxd3_48 worldly-wildflower-52"
    "foxd3_control_36 sunny-grass-49"
    "epha4a_control_24 genial-firefly-50"
    "phox2a_control_36 fanciful-river-48"
    "wnt3a_wnt8_18 celestial-shadow-44"
    "egr2b_control_18 legendary-moon-45"
    "wnt3a_wnt8_24 morning-paper-45"
    "met_control_72 confused-lake-45"
    "foxd3_control_72 frosty-darkness-43"
    "tbx16_msgn1_36 comic-bush-42"
    "tbx1_control_48 northern-serenity-41"
    "phox2a_control_48 neat-violet-40"
    "egr2b_control_48 toasty-bush-39"
    "hand2_control_48 likely-feather-38"
    "tfap2a_control_48 fearless-pine-37"
    "mafba_control_48 comic-feather-35"
    "noto_control_48 super-shape-35"
    "foxi1_control_48 twilight-yogurt-34"
    "smo_control_24 unique-energy-33"
    "wnt3a_wnt8_36 balmy-monkey-32"
    "zc4h2_control_48 distinctive-water-31"
    "tbx1_control_36 scarlet-snow-30"
    "mafba_control_24 sweet-sky-29"
    "tbx16_control_18 classic-lake-28"
    "mafba_control_18 peach-serenity-26"
    "hgfa_control_72 fast-glade-27"
    "tbx16_tbx16l_24 fragrant-salad-25"
    "smo_control_36 misunderstood-vortex-24"
    "tfap2a_foxd3_36 legendary-fire-23"
    "hoxb1a_control_48 clear-morning-22"
    "zc4h2_control_36 desert-moon-21"
    "cdx4_control_36 worthy-grass-20"
    "egr2b_control_24 breezy-glitter-19"
    "foxi1_control_72 neat-snowball-18"
    "tfap2a_control_36 clear-snowflake-17"
    "cdx4_control_18 ethereal-wildflower-16"
    "egr2b_control_36 smart-energy-15"
    "noto_control_36 electric-firefly-14"
    "tbx16_msgn1_24 earnest-dragon-11"
    "smo_control_18 astral-lake-8"
    "mafba_control_36 jumping-universe-11"
    "noto_control_24 divine-lion-9"
    "hand2_control_24 dutiful-sunset-9"
    "epha4a_control_48 deft-resonance-13"
    "cdx4_cdx1a_36 lucky-voice-7"
    "noto_control_18 earthy-dream-6"
    "tfap2a_foxd3_72 iconic-pyramid-5"
    "hgfa_control_48 treasured-sea-4"
    "tfap2a_control_72 apricot-wood-3"
    "zc4h2_control_24 swept-aardvark-2"
    "met_control_36 astral-haze-1"
)

# Loop through each tuple and submit a job
for tuple in "${tuples[@]}"; do
    # Read ood_cond and model_name from the tuple
    read -r ood_cond model_name <<< "$tuple"
    
    # Construct a SLURM job script and submit
    sbatch <<EOF
#!/bin/bash
#SBATCH -o ct_prop_"$model_name".out
#SBATCH -e ct_prop_"$model_name".err
#SBATCH -J ct_prop_"$model_name"
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -t 0-12:00:00
#SBATCH --nice=1

# Load required modules (if any)
source ${HOME}/.bashrc_new
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/cfp


# Print info
echo "Running on: \$(hostname)"
echo "ood_cond: $ood_cond, Model: $model_name"

# Run your program with the tuple as an argument
python compute_logfold_changes_mean_model.py "$ood_cond"
EOF
done
