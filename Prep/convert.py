import logging
import os
from dataclasses import dataclass
from typing import Annotated, Literal

import tyro

from utils.converter import process_cameras, process_images, process_flame, CustomFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

@dataclass
class Config:
    """
    Convert Lumio3D output to a dataset that is
    compatible with VHAP.

    INPUT File structure:

    --- input_dir/
        --- cameras/ (directly obtained from Lumio3D)
            --- cam_0.json
            --- cam_1.json
            ...
        --- data/ (directly optained from Lumio3D)
            --- cam_0/
                --- diff.exr
                --- flash_0.exr
                --- flash_1.exr
                --- diff_normal.exr
                --- mask.png
            --- cam_1/
                ...
    """

    input_dir: Annotated[str, tyro.conf.arg(aliases=["-i"])]
    """Input directory containing Lumio3D output."""
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])]
    """Output directory for the converted dataset."""
    width: int = 435
    """Width of the output images."""
    height: int = 574
    """Height of the output images."""
    image_type: Annotated[
        Literal["diff", "flash_0", "flash_1", "diff_normal"],
        tyro.conf.arg(aliases=["-itype"]),
    ] = "flash_1"
    """Type of input images to convert."""


def convert(input_dir, out_dir, width, height, image_type):
    logger.info(f"Converting files in directory: {input_dir}")
    assert os.path.isdir(input_dir), f"Directory {input_dir} does not exist."

    logger.info(f"Image dimensions: {width}x{height}")
    camera_paths, data_paths = (
        os.path.join(input_dir, "cameras"),
        os.path.join(input_dir, "data"),
    )

    if os.path.exists(out_dir):
        logger.warning(
            f"Output directory {out_dir} already exists. This will override it."
        )
        # sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    cam_output_dir = os.path.join(out_dir, "camera_params")
    image_output_dir = os.path.join(out_dir, "images")
    alpha_output_dir = os.path.join(out_dir, "alpha_maps")
    # process camera files
    process_cameras(camera_paths, width, height, cam_output_dir, rot=[])
    process_images(
        data_paths,
        image_output_dir,
        alpha_output_dir,
        width,
        height,
        image_type=image_type,
    )
    process_flame(
        input_dir,
        out_dir,
    )

    logger.info("DONE!")


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    config = tyro.cli(Config)
    convert(
        input_dir=config.input_dir,
        out_dir=config.output_dir,
        width=config.width,
        height=config.height,
        image_type=config.image_type,
    )
