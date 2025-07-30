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
conda activate lumio
python Prep/convert.py -i /mnt/nas/sitt/demo/data_prep/lumio_scans/mj/ -o data/to_export/$OUT_NAME -itype flash_1
conda deactivate

# Prepare for Gaussian Avatar
echo "Preparing for Gaussian Avatar..."
conda activate accel
python Prep/toGA.py  --src data/to_export/$OUT_NAME/ --tgt data/to_ga/$OUT_NAME
conda deactivate

# Train

conda activate gaussian-avatars
echo "Training Gaussian Avatar..."
cd GaussianAvatars
python train.py --bind_to_mesh --port 60030 --interval_media 100 --depth --interval 1000 --sh_degree 3 -s ../data/to_ga/$OUT_NAME --bcull -m ../result/pipe/$GA_NAME
conda deactivate