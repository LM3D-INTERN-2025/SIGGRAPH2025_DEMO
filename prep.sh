#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

OUT_NAME="bbs_f1"
GA_NAME="ga_bbs_f1"

# Prepare for nerf exporting
echo "Preparing for nerf exporting..."
conda activate lumio
python Prep/convert.py -i /mnt/nas/sitt/demo/data_prep/lumio_scans/bird_blue_shirt/ -o data/to_export/$OUT_NAME -itype flash_1
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
python train.py --bind_to_mesh --port 60030 --interval_media 100 --depth --interval 1000 --sh_degree 3 -s ../data/to_ga/$OUT_NAME --bcull -m output/pipe/$GA_NAME
conda deactivate