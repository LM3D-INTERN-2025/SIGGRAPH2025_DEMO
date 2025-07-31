import numpy as np
import math
import json
import os

transforms = {
    "timestep_indices": [
    ],
    "camera_indices": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
    ]
}

camera_params = {}
w2c = {}
intrinsics = {}
sum_fx = 0.0
sum_fy = 0.0

rot_90 = np.array([[ 0, -1,  0,  0],
                   [ 1,  0,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])

rot_180 = np.array([[-1,  0,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0,  1,  0],
                    [ 0,  0,  0,  1]])

mirror = np.array([[-1,  0,  0,  0],
                   [ 0,  1,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])

to_gs = np.array([[ 1,  0,  0,  0],
                  [ 0, -1,  0,  0],
                  [ 0,  0, -1,  0],
                  [ 0,  0,  0,  1]])

# flip = {3}
# flip = {3,4,5,6,7,10,11,12,13,14} #shu
flip = {3,5,7,10,11,12,13} # bird_blue_shirt
# flip = {0,1,2,4,6,8,9,11,15} #girl1
frames = []
# flip2 = {4,6,14}
flip2 = {}

# path = "/mnt/nas/pim/cameras/shu_cameras"
path = "/mnt/nas/pim/new_scan_data/bird_blue_shirt/cameras"

camera_num = 16
full_num = 5
segment_x = 2
segment_y = 3
sz_x = 1024
sz_y = 1024
crop = True

transforms["timestep_indices"] = list(range(segment_x*segment_y + full_num))

for i in range(camera_num):
    for px in range(segment_x):
        for py in range(segment_y):
            # if i != 0:
            #     continue
            timestep_id = str(int(px)*10 + int(py))
            timestep_id = '0'*(5-len(timestep_id)) + timestep_id
            timestep = px*segment_y + py
            
            # if int(timestep_id) not in transforms["timestep_indices"]:
            #     transforms["timestep_indices"].append(int(timestep_id))
            cam =str(i)
            cam = '0'*(2-len(cam)) + cam
            now_frame = {
                "timestep_index": int(timestep),
                "timestep_index_original": int(timestep_id),
                "timestep_id": "frame_" + timestep_id,
                "camera_index": i,
                "camera_id": "camera"+cam,
                "file_path": "images/"+timestep_id+"_"+cam+".png",
                "fg_mask_path": "fg_masks/"+timestep_id+"_"+cam+".png",
                "flame_param_path": "flame_param/"+"00000"+".npz"
            }
            f = open(path+'/camera'+cam+'.txt')
            s = f.read()
            s = s.split('\n')
            # scale = 0.4
            scale = 1.0
            h, w       = [float (i) * scale for i in s[1].split()]
            # h, w       = [972.0, 736.0]
            fl_y, fl_x = [float (i) * scale for i in s[3].split()]
            cy, cx     = [float (i) * scale for i in s[5].split()]
            # cy, cx     = [972.0 / 2.0, 736.0 / 2.0]
            m          = np.array([[float(j) for j in i.split()] for i in s[12:16]])
            # sh = 574.0/h
            # sw = 435.0/w
            sh = sz_y/h
            sw = sz_x/w
            # sh = 1.0
            # sw = 1.0

            ### scale images
            if not crop:
                w    *= sw
                fl_x *= sw
                cx   *= sw
                h    *= sh
                fl_y *= sh
                cy   *= sh

            ### crop images
            if crop:
                # print("timestep_id:", px,py)
                # print(cx,cy)
                cx -= (px/(segment_x-1)) * (w - sz_x)
                cy -= ((segment_y - 1 - py)/(segment_y-1)) * (h - sz_y)
                # print(cx,cy)
                w *= sw
                h *= sh

            intr = np.array([[      fl_x,          0.0,          cx],
                            [       0.0,         fl_y,          cy],
                            [       0.0,          0.0,         1.0],])

            # m[:,0] *= -1.0
            # r = m[:3, :3].T
            # r[[0,1]] = r[[1,0]]
            # t = m[:3, 3] /100.0
            # t[2] *= -1.0
            r = np.identity(4)
            t = np.identity(4)
            t_off = np.identity(4)
            # t_off[3:, 2] = 5 #cy/fl_y
            t_off2 = t_off.copy()
            t_off2[3:, :3] *= -1.0
            r[:3, :3] = m[:3, :3]
            t[:3,  3] = m[:3,  3] * -1.0 * 0.01

            r = r @ rot_90 
            r = r @ rot_180
            if i in flip:
                r = r @ rot_180
            if i in flip2:
                r = r @ rot_180
            r = r @ mirror
            r = r.T
            r2 = r.copy()
            r2 = r2.T
            r2 = r2 @ to_gs
            t2 = t.copy()
            t2[:3, 3] *= -1.0
            m = r @ t#w2c
            m2 = t2 @ r2 #c2w
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            now_frame["transform_matrix"] = m2.tolist()
            now_frame["cx"] = cx
            now_frame["cy"] = cy
            now_frame["fl_x"] = fl_x
            now_frame["fl_y"] = fl_y
            now_frame["h"] = h
            now_frame["w"] = w
            now_frame["camera_angle_x"] = angle_x
            now_frame["camera_angle_y"] = angle_y

            frames.append(now_frame)

            if i == 0:
                camera_params["width"] = w
                camera_params["height"] = h
            sum_fx += fl_x
            sum_fy += fl_y
            w2c[str(i)] = m.tolist()
            intrinsics[str(i)] = intr.tolist()

    for j in range(full_num):
        timestep_id = str(int(j)*100)
        timestep_id = '0'*(5-len(timestep_id)) + timestep_id
        timestep = px*segment_y + py + j
        
        # if int(timestep_id) not in transforms["timestep_indices"]:
        #     transforms["timestep_indices"].append(int(timestep_id))
        cam =str(i)
        cam = '0'*(2-len(cam)) + cam
        now_frame = {
            "timestep_index": int(timestep),
            "timestep_index_original": int(timestep_id),
            "timestep_id": "frame_" + timestep_id,
            "camera_index": i,
            "camera_id": "camera"+cam,
            "file_path": "images/"+timestep_id+"_"+cam+".png",
            "fg_mask_path": "fg_masks/"+timestep_id+"_"+cam+".png",
            "flame_param_path": "flame_param/"+"00000"+".npz"
        }
        f = open(path+'/camera'+cam+'.txt')
        s = f.read()
        s = s.split('\n')
        scale = 0.4
        # scale = 1.0
        h, w       = [float (i) for i in s[1].split()]
        # h, w       = [972.0, 736.0]
        fl_y, fl_x = [float (i) for i in s[3].split()]
        cy, cx     = [float (i) for i in s[5].split()]
        # cy, cx     = [972.0 / 2.0, 736.0 / 2.0]
        m          = np.array([[float(j) for j in i.split()] for i in s[12:16]])
        # sh = 574.0/h
        # sw = 435.0/w
        # sh = 1.0
        # sw = 1.0

        ### scale images
        if not crop:
            w    *= scale
            fl_x *= scale
            cx   *= scale
            h    *= scale
            fl_y *= scale
            cy   *= scale

        intr = np.array([[      fl_x,          0.0,          cx],
                        [       0.0,         fl_y,          cy],
                        [       0.0,          0.0,         1.0],])

        r = np.identity(4)
        t = np.identity(4)

        r = r @ rot_90 
        r = r @ rot_180
        if i in flip:
            r = r @ rot_180
        if i in flip2:
            r = r @ rot_180
        r = r @ mirror
        r = r.T
        r2 = r.copy()
        r2 = r2.T
        r2 = r2 @ to_gs
        t2 = t.copy()
        t2[:3, 3] *= -1.0
        m = r @ t #w2c
        m2 = t2 @ r2 #c2w
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        now_frame["transform_matrix"] = m2.tolist()
        now_frame["cx"] = cx
        now_frame["cy"] = cy
        now_frame["fl_x"] = fl_x
        now_frame["fl_y"] = fl_y
        now_frame["h"] = h
        now_frame["w"] = w
        now_frame["camera_angle_x"] = angle_x
        now_frame["camera_angle_y"] = angle_y

        frames.append(now_frame)

        if i == 0:
            camera_params["width"] = w
            camera_params["height"] = h
        sum_fx += fl_x
        sum_fy += fl_y
        w2c[str(i)] = m.tolist()
        intrinsics[str(i)] = intr.tolist()

transforms["frames"] = frames
transforms["cx"] = transforms["frames"][0]["cx"]
transforms["cy"] = transforms["frames"][0]["cy"]
transforms["fl_x"] = transforms["frames"][0]["fl_x"]
transforms["fl_y"] = transforms["frames"][0]["fl_y"]
transforms["h"] = transforms["frames"][0]["h"]
transforms["w"] = transforms["frames"][0]["w"]
transforms["camera_angle_x"] = transforms["frames"][0]["camera_angle_x"]
transforms["camera_angle_y"] = transforms["frames"][0]["camera_angle_y"]
# print(transforms)

camera_params["world_2_cam"] = w2c
# camera_params["intrinsics"] = [[ sum_fx / 16.0,           0.0, camera_params["width"]  / 2.0],
#                                [           0.0, sum_fy / 16.0, camera_params["height"] / 2.0],
#                                [           0.0,           0.0,                           1.0]]
camera_params["intrinsics"] = intrinsics

json_object = json.dumps(transforms, indent=4)
# print(json_object)
path = "/mnt/nas/pim/export/lumio_sample/bird_v06/"
try:
    os.remove(path + "transforms.json")
    os.remove(path + "transforms_backup_flame.json")
    os.remove(path + "transforms_backup.json")
    os.remove(path + "transforms_val.json")
    os.remove(path + "transforms_train.json")
    os.remove(path + "transforms_test.json")
except:
    print("ok ja")

# with open("camera.json", "x") as outfile:
#     outfile.write(json_object)

with open(path + "transforms.json", "x") as outfile:
    outfile.write(json_object)
with open(path + "transforms_backup_flame.json", "x") as outfile:
    outfile.write(json_object)
with open(path + "transforms_backup.json", "x") as outfile:
    outfile.write(json_object)

# transforms["frames"] = frames[4:5]
# json_object = json.dumps(transforms, indent=4)
with open(path + "transforms_val.json", "x") as outfile:
    outfile.write(json_object)

# transforms["frames"] = frames[:4]+frames[5:]
# json_object = json.dumps(transforms, indent=4)
with open(path + "transforms_train.json", "x") as outfile:
    outfile.write(json_object)
with open(path + "transforms_test.json", "x") as outfile:
    outfile.write(json_object)

#### VHAP
# json_object = json.dumps(camera_params, indent=4)
# path = "/mnt/nas/pim/lumio_sample/camera_params/bird_v02"
# try:
#     os.remove(path + "/camera_params.json")
# except:
#     print("ok ngub")

# with open(path + "/camera_params.json", "x") as outfile:
#     outfile.write(json_object)