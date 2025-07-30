#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# let user input the name of the output
echo "Please enter the name of the output (default: date_time):"
read OUT_NAME
if [ -z "$OUT_NAME" ]; then
    echo "No name provided, using default: date_time"
    OUT_NAME="$(date +%Y%m%d_%H%M%S)"
fi
GA_NAME="ga_$OUT_NAME"

# Prepare for nerf exporting
echo "Preparing for nerf exporting..."
conda activate accel
python Prep/convert.py -i /mnt/nas/sitt/demo/data_prep/lumio_scans/dekdee -o Data/to_export/$OUT_NAME -itype diff

# Prepare for Gaussian Avatar
echo "Preparing for Gaussian Avatar..."
python Prep/toGA.py  --src Data/to_export/$OUT_NAME/ --tgt Data/to_ga/$OUT_NAME
conda deactivate

# Train

# random port number
RANDOM_PORT=$((RANDOM % 10000 + 60000))

conda activate gaussian-avatars
echo "Training Gaussian Avatar..."
echo "Using port: $RANDOM_PORT"
cd GaussianAvatars
python train.py --bind_to_mesh --port $RANDOM_PORT --interval_media 100 --depth --interval 1000 --sh_degree 0 -s ../Data/to_ga/$OUT_NAME --bcull -m ../Result/$GA_NAME
conda deactivate