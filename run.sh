#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# let user input the path to the data
echo "Please enter the input data folder name :"
read INPUT_DATA
if [ -z "$INPUT_DATA" ]; then
    echo "No input data provided, Aborting..."
    exit 1
fi 

# let user input the name of the output
echo "Please enter the name of the output (default: date_time):"
read OUT_NAME
if [ -z "$OUT_NAME" ]; then
    OUT_NAME="$(date +%Y%m%d_%H%M%S)"
    echo "No name provided, using default: $OUT_NAME"
fi
GA_NAME="ga_$OUT_NAME"

# let user input the type of image
echo "Please enter the type of image from [diff, flash_0, flash_1, flash_2] (default: flash_1):"
read IMAGE_TYPE
if [ -z "$IMAGE_TYPE" ]; then
    IMAGE_TYPE="flash_1"
    echo "No image type provided, using default: $IMAGE_TYPE"
fi

# let user input the port number
echo "Please enter the training port number (default: 60030):"
read RANDOM_PORT
if [ -z "$RANDOM_PORT" ]; then
    RANDOM_PORT=60030
    echo "No port number provided, using default: $RANDOM_PORT"
fi

conda activate ga

# prepare for nerf exporting
echo "Preparing for nerf exporting..."
python Prep/convert.py -i Data/$INPUT_DATA --width 1740 --height 2296 -o Data/to_export/$OUT_NAME -itype flash_1

# prepare for Gaussian Avatar
echo "Preparing for Gaussian Avatar..."
python Prep/toGA.py  --src Data/to_export/$OUT_NAME/ --tgt Data/to_ga/$OUT_NAME

# train
echo "Training Gaussian Avatar..."
cd GaussianAvatars
python train.py --bind_to_mesh --port $RANDOM_PORT --interval_media 100 --depth --interval 1000 --sh_degree 3 -s ../Data/to_ga/$OUT_NAME --bcull -m ../Result/$GA_NAME

conda deactivate