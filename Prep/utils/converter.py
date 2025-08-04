import json
import os
import sys
from PIL import Image
import numpy as np
import cv2
import logging
import re

from utils.config import rot_dict, train_cam
from utils.camera import construct_intrinsics

from pathlib import Path

from tqdm import tqdm


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format = "%(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# ---------------------- CAMERAS STUFFs ----------------------


def swap_columns(matrix, col1, col2):
    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
    return matrix


def swap_rows(matrix, row1, row2):
    matrix[[row1, row2], :] = matrix[[row2, row1], :]
    return matrix


def parse_camera_matrix_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    matrix_lines = []
    reading_matrix = False

    for line in lines:
        line = line.strip()

        if line.startswith("MATRIX") or line.endswith("MATRIX :"):
            reading_matrix = True
            continue

        if reading_matrix:
            if line == "":
                break
            row_values = list(map(float, line.split()))
            matrix_lines.append(row_values)

            if len(matrix_lines) == 4:
                break

    if len(matrix_lines) != 4:
        raise ValueError("Matrix format is invalid or incomplete")

    return np.array(matrix_lines)


def parse_intrinsics_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    values = []

    reading_focal = False

    for line in lines:
        line = line.strip()
        if line.startswith("#focal"):
            reading_focal = True
            continue
        if reading_focal:
            if line == "":
                break
            focal_x = float(line.split()[0])
            focal_y = float(line.split()[1])
            values.append(focal_x)
            values.append(focal_y)
            reading_focal = False
            continue

    if len(values) != 2:
        raise ValueError("Intrinsics format is invalid or incomplete")

    return np.array(values)


def process_cameras(camera_dir, width, height, output_dir, rot=[]):
    rot = [idx for idx, _ in rot_dict.items() if _ == 90]

    matrices = []
    intrinsics = []
    for name in os.listdir(camera_dir):
        if name.endswith(".txt"):
            file_path = os.path.join(camera_dir, name)
            try:
                camera_matrix = parse_camera_matrix_from_file(file_path)
                intrinsics_value = parse_intrinsics_from_file(file_path)
                cam_num = int(name.split(".")[0][-2:])
                if cam_num in train_cam:
                    matrices.append((cam_num, camera_matrix))
                    intrinsics.append((cam_num, intrinsics_value))
            except Exception as e:
                logger.error(f"Error parsing {name}: {e}", file=sys.stderr)

    matrices.sort(key=lambda x: x[0])
    intrinsics.sort(key=lambda x: x[0])

    #  ------------------------ INTRINSICS ------------------------
    scaling_factor_x = width / 1842
    scaling_factor_y = height / 2432

    image_transform = np.array(
        [
            [scaling_factor_x, 0, scaling_factor_x * 0.5 - 0.5],
            [0, scaling_factor_y, scaling_factor_y * 0.5 - 0.5],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    out_intrins = []
    for cam_num, intrinsics_value in intrinsics:
        _intrinsic = construct_intrinsics(
            intrinsics_value[0],
            intrinsics_value[1],
            921.000000,
            1216.000000,
        )
        new_matrix = image_transform @ _intrinsic
        out_intrins.append((cam_num, new_matrix))

    # ------------------------ EXTRINSICS ------------------------
    out_extr = []

    for m in matrices:
        cam = m[0]

        # mat should be cam2world
        mat = m[1].copy()

        # rotate the cam 90 deg around z (do it in cam coord)
        swap_columns(mat, 0, 1)

        # inverse -> world2cam
        mat = np.linalg.inv(mat)

        # scale the translation (cm -> m)
        mat[:3, 3] /= 100

        # arbitrary rotation (?)
        if cam in rot:
            mat[[0, 1], :] *= -1

        out_extr.append((cam, mat))

    # ------------------------ OUTPUT ------------------------

    logger.info(f"Writing camera parameters to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "camera_params.json")

    with open(output_file, "w+") as f:
        json.dump(
            {
                "intrinsics": {
                    cam: mat.tolist() for cam, mat in out_intrins
                },
                "world_2_cam": {cam: mat.tolist() for cam, mat in out_extr},
                "width": width,
                "height": height,
            },
            f,
            indent=4,
        )

    # logger.info(f"Camera data written to {output_file}")


# ---------------------- IMAGE CONVERSION ----------------------

# IMAGE MAGICK CONVERTER
# def converter(input_file, output_file, w, h, rot=0):
#     command = f"convert {input_file} -rotate {rot} -resize {w}x{h}! {output_file}"
#     os.system(command)
#     logger.info(f"Converted {input_file} to {output_file}")


# PYTHON CONVERTER
def converter(input_file, output_file, w, h, rot=0):
    img_ = cv2.imread(input_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    if input_file.endswith(".exr"):
        img_ = img_ ** (1 / 2.2)  # gamma correction
    if input_file.endswith(".png"):
        img_ = img_[:, :, :1].squeeze()  # (w, h, 3) -> (w, h)

    if img_.dtype != np.uint8:
        # img_ = np.clip(img_, 0, 1) if img_.dtype == np.float32 else img_ / 65535.0
        img_ = (img_ * 255).astype(np.uint8)

    # logger.info(f"{input_file} shape: {img_.shape}, dtype: {img_.dtype}")
    img = Image.fromarray(img_)
    if rot != 0:
        img = img.rotate(-rot, expand=True)
    img = img.resize(
        (w, h), Image.Resampling.LANCZOS
    )  # could try other resampling algo

    img.save(output_file, quality=95)
    # logger.info(f"Converted {input_file} to {output_file}")


def alphanumeric(data):  # natural sort
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)


def process_images(
    data_dir, output_dir, alpha_out_dir, width, height, image_type="diff"
):
    for cam in tqdm(alphanumeric(os.listdir(data_dir)), desc="Processing images"):
        # logger.info(f"Processing file: {cam}")
        for file in os.listdir(os.path.join(data_dir, cam)):
            if file.endswith(f"{image_type}.exr"):
                input_file = os.path.join(data_dir, cam, file)
                file_name = f"cam_{cam[3:]}_000000.jpg"
                output_file = os.path.join(output_dir, file_name)
                if not os.path.exists(os.path.join(output_dir)):
                    os.makedirs(output_dir)
                converter(
                    input_file, output_file, width, height, rot=rot_dict[int(cam[3:])]
                )
            if file.endswith("mask.png"):
                input_file = os.path.join(data_dir, cam, file)
                file_name = f"cam_{cam[3:]}_000000.jpg"
                output_file = os.path.join(alpha_out_dir, file_name)
                if not os.path.exists(os.path.join(alpha_out_dir)):
                    os.makedirs(alpha_out_dir)
                converter(
                    input_file, output_file, width, height, rot=rot_dict[int(cam[3:])]
                )


def convert_flame(path, static_offset_path, out_path):

    s = open(path).read().split('\n')
    s = [line.strip() for line in s if line.strip()]

    print(len(s))
    print(len(s[0].split(' ')))
    print(len(s[1].split(' ')))
    print(len(s[2].split(' ')))
    shape = np.array([float(i) for i in list(filter(lambda x: len(x)!=0, s[0].split(' ')[:300]))])
    expr  = np.array([float(i) for i in list(filter(lambda x: len(x)!=0, s[0].split(' ')[300:]))]).reshape(1,100)
    pose  = np.array([float(i) for i in list(filter(lambda x: len(x)!=0, s[1].split(' ')))])
    tran  = np.array([float(i) for i in list(filter(lambda x: len(x)!=0, s[2].split(' ')))]).reshape(1,3)
    print(len(shape),len(expr),len(pose),len(tran))

    pose *= -1
    expr *= -1

    rot  = pose[ :3].reshape(1,3)
    neck = pose[3:6].reshape(1,3)
    jaw  = pose[6:9].reshape(1,3)
    eye  = pose[9: ].reshape(1,6)
    
    # static offsets
    static_offset = np.loadtxt(static_offset_path, dtype=np.float32)
    static_offset = np.concatenate([static_offset, np.zeros((5143 - 5023, 3))], axis=0)
    static_offset = static_offset.reshape(1, 5143, 3)
    static_offset /= 100 # cm to m

    # offset = np.zeros((1,5143,3))

    np.savez(out_path,translation = tran ,rotation = rot ,neck_pose = neck,jaw_pose = jaw ,eyes_pose = eye ,shape = shape ,expr = expr, static_offset = static_offset)

def process_flame(
    data_dir, output_dir
):
    result_params_file = None
    static_offset_file = None
    for filepath in Path(data_dir).rglob("*.txt"):
        if "resultParams.txt" in filepath.name:
            logger.info(f"Found resultParams.txt: {filepath}")
            result_params_file = filepath
        if "offsets.txt" in filepath.name:
            logger.info(f"Found offsets.txt: {filepath}")
            static_offset_file = filepath
        if result_params_file and static_offset_file:
            break
    if result_params_file is None:
        logger.error("resultParams.txt not found in the data directory.")
        return
    if static_offset_file is None:
        logger.error("offsets.txt not found in the data directory.")
        return

    logger.info(f"Processing FLAME parameters...")

    convert_flame(result_params_file, static_offset_file, os.path.join(output_dir, "flame_params.npz"))
    


if __name__ == "__main__":
    pass
