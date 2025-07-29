import os
import pathlib
from pathlib import Path
from copy import deepcopy
from typing import Optional
import numpy as np
import PIL.Image as Image
import torch
import json

from torch.utils.data import Dataset

import torchvision.transforms.functional as F


from utils import camera

class LumioDataset(Dataset):
    def __init__(
            self,
            source_path: pathlib.Path,
        ):
        super().__init__()
        self.source_path = source_path
        self.target_extrinsic_type = "c2w"  # or "w2c"
        self.camera_convention_conversion = "opencv->opengl"

        self.use_alpha_map = True
        self.background_color = None  # white, black
        self.img_to_tensor = False

        self.define_properties()
        self.load_cameras()

        self.timestep_ids = set(
            f.split('.')[0].split('_')[-1]
            for f in os.listdir(self.source_path / self.properties['rgb']['folder']) if f.endswith(self.properties['rgb']['suffix'])
        )
        self.timestep_ids = sorted(self.timestep_ids) # ['000000', '000001', '000002', '000003', ....]
        self.timestep_indices = list(range(len(self.timestep_ids)))

        print(f"number of timesteps: {self.num_timesteps}, number of cameras: {self.num_cameras}")

        self.items = []
        for fi, timestep_index in enumerate(self.timestep_indices):
            for ci, camera_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": fi,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": self.timestep_ids[timestep_index],
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )
        print(self.items[0])

    @property
    def num_timesteps(self):
        return len(self.timestep_indices)

    @property
    def num_cameras(self):
        return len(self.camera_ids)

    def define_properties(self):
        self.properties = {
            "rgb": {
                "folder": "images",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": "jpg",
            },
            "alpha_map": {
                "folder": "alpha_maps",
                "cam_id_prefix": "cam_",
                "per_timestep": True,
                "suffix": "jpg",
            },
        }
    def load_cameras(self, camera_params_path: Optional[pathlib.Path] = None):
        
        if camera_params_path is None:
            camera_params_path = self.source_path / "camera_params" / "camera_params.json"

        assert camera_params_path.exists()
        param = json.load(open(camera_params_path))

        self.camera_ids =  list(param["world_2_cam"].keys())

        K = torch.tensor([param["intrinsics"][k] for k in self.camera_ids])  # (N, 3, 3)

        if "height" not in param or "width" not in param:
            raise ValueError("Camera parameters must contain 'height' and 'width' keys.")
        else:
            H, W = param["height"], param["width"]

        w2c = torch.tensor([param["world_2_cam"][k] for k in self.camera_ids])  # (N, 4, 4)
        R = w2c[..., :3, :3]
        T = w2c[..., :3, 3]

        orientation = R.transpose(-1, -2)  # (N, 3, 3)
        location = R.transpose(-1, -2) @ -T[..., None]  # (N, 3, 1)

        orientation, K = camera.convert_camera_convention(
                self.camera_convention_conversion, orientation, K, H, W
            )

        c2w = torch.cat([orientation, location], dim=-1)  # camera-to-world transformation

        if self.target_extrinsic_type == "w2c":
            R = orientation.transpose(-1, -2)
            T = orientation.transpose(-1, -2) @ -location
            w2c = torch.cat([R, T], dim=-1)  # world-to-camera transformation
            extrinsic = w2c
        elif self.target_extrinsic_type == "c2w":
            extrinsic = c2w
        else:
            raise NotImplementedError(f"Unknown extrinsic type: {self.target_extrinsic_type}")

        # print("W, H:", W, H)
        # print("intrinsic:", K)
        # print("extrinsic:", extrinsic)
        print("Successfully loaded camera parameters from", camera_params_path)
        self.camera_params = {}
        for i, camera_id in enumerate(self.camera_ids):
            self.camera_params[camera_id] = {"intrinsic": K[i], "extrinsic": extrinsic[i]}


    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        folder = p["folder"] if "folder" in p else None
        per_timestep = p["per_timestep"]
        suffix = p["suffix"]

        path = self.source_path
        if folder is not None:
            path = path / folder

        if self.num_cameras > 1:
            if camera_id is None:
                assert (
                    index is not None), "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_prefix" in p:
                camera_id = p["cam_id_prefix"] + camera_id
        else:
            camera_id = ""

        if per_timestep:
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            if len(camera_id) > 0:
                path /= f"{camera_id}_{timestep_id}.{suffix}"
            else:
                path /= f"{timestep_id}.{suffix}"
        else:
            if len(camera_id) > 0:
                path /= f"{camera_id}.{suffix}"
            else:
                path = Path(str(path) + f".{suffix}")

        return path

    def apply_transforms(self, item):

        if self.background_color is not None:
            assert (
                "alpha_map" in item
            ), "'alpha_map' is required to apply background color."
            fg = item["rgb"]
            if self.background_color == "white":
                bg = np.ones_like(fg) * 255
            elif self.background_color == "black":
                bg = np.zeros_like(fg)
            else:
                raise NotImplementedError(
                    f"Unknown background color: {self.background_color}."
                )
            w = item["alpha_map"][..., None] / 255
            img = (w * fg + (1 - w) * bg).astype(np.uint8)
            item["rgb"] = img

        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])
            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])


        return item


    def __len__(self):
        return len(self.camera_ids)
    
    def __getitem__(self, idx):
        item = deepcopy(self.items[idx])

        rgb_path = self.get_property_path("rgb", idx)
        item["rgb"] = np.array(Image.open(rgb_path))

        camera_param = self.camera_params[item["camera_id"]]
        item["intrinsic"] = camera_param["intrinsic"].clone()
        item["extrinsic"] = camera_param["extrinsic"].clone()

        if self.use_alpha_map or self.background_color is not None:
            alpha_path = self.get_property_path("alpha_map", idx)
            item["alpha_map"] = np.array(Image.open(alpha_path))

        item = self.apply_transforms(item)
        return item