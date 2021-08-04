import tensorflow as tf
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
from SDDS_Models import MP_CNN, strided_Conv_CNN, SDDS_Net, SDDS_Net_DW
import matplotlib.pyplot as plt
import random
import math
import time

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    return abs_rel, rmse, a1, a2, a3
    
#load SDDS-Net model with benchmark options of 'NYUV2', set sim to True if want simultaneous task
model = SDDS_Net(benchmark= 'NYUV2', sim = False, weights_path='')
model.summary()

sim = False  #set to True to enable simultaneous task of depth estimation and semantic segmentation
depth = False   # False mean semantic segmentation task and True means depth estimation task

depth_path = 'E:/NYU depthv2/depth/'

with open('val.txt','r') as f:
    val_data = f.readlines()
val_list = []
errors = []
for data in val_data:
    m = tf.keras.metrics.MeanIoU(num_classes=14)
    old = time.time()
    img_name = str(int(data.split('_')[-1].split('.png')[0]))+'.png'
    image = cv2.imread('E:/NYUv2 semantic segmentation/images/'+str(int(data.split('_')[-1].split('.png')[0]))+'.png')
    image = cv2.resize(image, (640,480))
    img = image/255.
    img = np.expand_dims(img, axis = 0)
    if sim:
        #predict segmentation and depth maps 
        seg_map, depth_map = model.predict(img)
        seg_map = seg_map[0,:,:,0]
        depth_map = depth_map[0,:,:,0]
        now = time.time()
        seg_map = np.rint(seg_map)
        seg_map[seg_map<0] = 0
        seg_map[seg_map>13] = 13
        
        print(now-old)
        #create gt and prediction masks to split each mask separately for evaluation with gt
        sparse_seg_map = np.zeros((480,640,14))
        sparse_seg = np.zeros((480,640,14))
        for i in range(14):
            sparse_seg_map[seg_map == i, i] = 1
        seg = cv2.imread('E:/NYUv2 semantic segmentation/images_seg/'+data.split('\n')[0], cv2.IMREAD_GRAYSCALE)
        for i in range(14):
            sparse_seg[seg == i, i] = 1
        #load depth mask from .npy file 
        depth = np.load(depth_path+str(int(data.split('_')[-1].split('.png')[0]))+'.npy')*10.0
        #update mIOU value 
        m.update_state(sparse_seg, sparse_seg_map) 
        miou_val = m.result().numpy()
        val_list.append(miou_val)
        #evaluate with valid pixels only
        ind = np.where(depth == 0)
        depth2 = depth
        depth2[depth2 == 0] = 1
        depth_map[ind] = 1
        errors.append(compute_errors(depth, depth_map))
    else:
        if depth:
            #predict depth map
            depth_map = model.predict(img)
            depth_map = depth_map[0,:,:,0]
            #load depth mask from .npy file 
            depth = np.load(depth_path+str(int(data.split('_')[-1].split('.png')[0]))+'.npy')*10.0
            #evaluate with valid pixels only
            ind = np.where(depth == 0)
            depth2 = depth
            depth2[depth2 == 0] = 1
            depth_map[ind] = 1
            errors.append(compute_errors(depth, depth_map))
        else:
            #predict segmentation map
            seg_map = model.predict(img)
            seg_map = seg_map[0,:,:,0]
            now = time.time()
            seg_map = np.rint(seg_map)
            seg_map[seg_map<0] = 0
            seg_map[seg_map>13] = 13
            #create gt and prediction masks to split each mask separately for evaluation with gt
            sparse_seg_map = np.zeros((480,640,14))
            sparse_seg = np.zeros((480,640,14))
            for i in range(14):
                sparse_seg_map[seg_map == i, i] = 1
            seg = cv2.imread('E:/NYUv2 semantic segmentation/images_seg/'+data.split('\n')[0], cv2.IMREAD_GRAYSCALE)
            for i in range(14):
                sparse_seg[seg == i, i] = 1
            #update mIOU value
            m.update_state(sparse_seg, sparse_seg_map) 
            miou_val = m.result().numpy()
            val_list.append(miou_val)
        
print('mean IOU: ', sum(val_list)/len(val_list))
mean_errors = np.array(errors).mean(0)  
print('mean depth errors: ',mean_errors)