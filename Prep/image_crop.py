import os
import shutil
import cv2
import numpy as np
# /mnt/nas/pim/normal_corr_faces/shu/cam0
input_dir_path = "/mnt/nas/pim/new_scan_data/bird_blue_shirt/data"
output_dir_path = "/mnt/nas/pim/export/lumio_sample/bird_v02"
vhap_dir_path = "/mnt/nas/pim/lumio_sample/bird_v02/seq1"

try:
    shutil.rmtree(output_dir_path+"/fg_masks")
    shutil.rmtree(output_dir_path+"/images")
    shutil.rmtree(vhap_dir_path+"/alpha_maps")
    shutil.rmtree(vhap_dir_path+"/images")
except:
    print("ok")

os.mkdir(output_dir_path+"/fg_masks")
os.mkdir(output_dir_path+"/images")
os.mkdir(vhap_dir_path+"/alpha_maps")
os.mkdir(vhap_dir_path+"/images")
# try:
#     shutil.rmtree(output_dir_path)
#     shutil.rmtree(vhap_dir_path)
# except:
#     print("ok")
# os.mkdir(output_dir_path)
# os.mkdir(output_dir_path+"/fg_masks")
# os.mkdir(output_dir_path+"/images")
# os.mkdir(vhap_dir_path)
# os.mkdir(vhap_dir_path+"/alpha_maps")
# os.mkdir(vhap_dir_path+"/images")

dir_list = os.listdir(input_dir_path)
# print(dir_list)
# flip = {3,4,5,6,7,10,11,12,13,14} #shu
# flip = {0,1,2,4,6,8,9,11,15} #girl1
flip = {3,5,7,10,11,12,13}

segment_x = 3
segment_y = 2
sz_x = 1024
sz_y = 1024

for cam in dir_list:
    print("process : ", input_dir_path+"/"+cam+" ...")
    n = cam[3:]
    nn = '0'*(2-len(n)) + n
    for px in range(segment_x):
        for py in range(segment_y):
            timestep = str(int(py)*10 + int(px))
            timestep = '0'*(5-len(timestep)) + timestep
            print("timestep:",timestep)
            mask_img_path = input_dir_path+'/'+cam+'/mask.png'
            alpha_tmp_path = output_dir_path+'/fg_masks/tmp_'+nn+'.jpg'
            cmd = "convert " + mask_img_path + " " + alpha_tmp_path
            # print("-->",cmd)
            os.system(cmd)
            alpha_tmp_img = cv2.imread(alpha_tmp_path)
            h, w, channels = alpha_tmp_img.shape
            if int(cam[3:]) in flip: 
                print ("flip mask", nn)
                alpha_tmp_img = cv2.rotate(alpha_tmp_img, cv2.ROTATE_180)
            ### scale images:
            # alpha_png = cv2.resize(alpha_tmp_img, (sz_x,sz_y))
            # alpha_png = cv2.resize(alpha_tmp_img, (int(w*scale_ratio),int(h*scale_ratio)))
            ### crop images:
            start_x = (w-sz_x) * px // (segment_x-1)
            start_y = (h-sz_y) * py // (segment_y-1)
            alpha_png = alpha_tmp_img[start_y:start_y + sz_y, start_x:start_x + sz_x]
            alpha_png = cv2.rotate(alpha_png, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(vhap_dir_path+'/alpha_maps/cam_'+n+'_0'+timestep+'.png', alpha_png)
            cv2.imwrite(output_dir_path+'/fg_masks/'+timestep+'_'+nn+'.png', alpha_png)
            # cv2.imwrite(output_dir_path+'/fg_masks/cam_'+nn+'_000000.png', alpha_png)
            os.remove(alpha_tmp_path)

            exr_path = input_dir_path+'/'+cam+'/flash_1.exr'
            jpg_tmp_path = output_dir_path+'/images/tmp_'+nn+'.jpg'
            cmd = "convert " + exr_path + " " + jpg_tmp_path
            # print("-->",cmd)
            os.system(cmd)
            jpg_tmp_img = cv2.imread(jpg_tmp_path)
            if int(cam[3:]) in flip: 
                print("flip img", nn)
                jpg_tmp_img = cv2.rotate(jpg_tmp_img, cv2.ROTATE_180)
            ### scale images
            # img_png = cv2.resize(jpg_tmp_img, (574,435))
            # img_png = cv2.resize(jpg_tmp_img, (int(w*scale_ratio),int(h*scale_ratio)))
            ### crop images
            img_png = jpg_tmp_img[start_y:start_y + sz_y, start_x:start_x + sz_x]
            img_png = cv2.rotate(img_png, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # print ("debug:", int(nn) in flip, str(nn), flip)
            cv2.imwrite(vhap_dir_path+'/images/cam_'+n+'_0'+timestep+'.png', img_png)
            img_png = np.append(img_png, alpha_png[: ,: , :1],axis = 2)
            cv2.imwrite(output_dir_path+'/images/'+timestep+'_'+nn+'.png', img_png)
            os.remove(jpg_tmp_path)
    # break