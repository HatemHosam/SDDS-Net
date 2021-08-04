import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
import os
import cv2
import numpy as np
from SDDS_Models import MP_CNN, strided_Conv_CNN, SDDS_Net, SDDS_Net_DW
import matplotlib.pyplot as plt
import math
import time

errors = []

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

images_path = 'E:/cityscapes/DTS_Net/leftImg8bit_trainvaltest/leftImg8bit/'
test_anno_path = 'E:/cityscapes/DTS_Net/disparity_trainvaltest/disparity/val/'

sim = False  #set to True to enable simultaneous task of depth estimation and semantic segmentation
depth = False   # False mean semantic segmentation task and True means depth estimation task

#load SDDS-Net model with benchmark options of 'CITYSCAPES', set sim to True if you want simultaneous task
model = SDDS_Net(benchmark= 'CITYSCAPES', sim = False, weights_path='')
model.summary()

anno_test = []
test_data = os.listdir(test_anno_path)
for data in test_data:
    list = os.listdir(test_anno_path+data)
    for file in list:
        anno_test.append(test_anno_path+data+'/'+file)

train_w , train_h = 1024, 512

val_data = []
folders = os.listdir('E:/cityscapes/DTS_Net/leftImg8bit_trainvaltest/leftImg8bit/val/')
for folder in folders:
    img_list = os.listdir('E:/cityscapes/DTS_Net/leftImg8bit_trainvaltest/leftImg8bit/val/'+folder)
    for img_name in img_list:
        val_data.append(folder+'/'+img_name)
errors = []
val_list = []

for data in val_data:
    m = tf.keras.metrics.MeanIoU(num_classes=34)
    old = time.time()
    image = cv2.imread('E:/cityscapes/DTS_Net/leftImg8bit_trainvaltest/leftImg8bit/val/'+data)
    image = cv2.resize(image, (1024,512))
    img = image/255.
    img = np.expand_dims(img, axis = 0)
    if sim:
        seg_map, depth_map = model.predict(img)
        seg_map = seg_map[0,:,:,0]
        depth_map = depth_map[0,:,:,0]
    else:
        if depth:
            depth_map = model.predict(img)
        else:
            seg_map = seg_map[0,:,:,0]
            class_num = 33
            seg_map[seg_map<0] = 0
            seg_map[seg_map>33] = 33
            seg_map = np.rint(seg_map)
    now = time.time()
    print(now-old)
    if sim:
        depth = cv2.imread('E:/cityscapes/DTS_Net/disparity_trainvaltest/disparity/val/'+data.replace('leftImg8bit.png','disparity.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = cv2.resize(depth, dsize=(1024,512), interpolation=cv2.INTER_NEAREST)
        depth[depth > 0] = (depth[depth > 0] - 1) / 256
    
        pred = depth_map
        gt = depth
        ind = np.where(depth == 0)
        gt[gt == 0] = 1
        pred[ind] = 1
        errors.append(compute_errors(gt, pred))
        seg = cv2.imread('E:/cityscapes/DTS_Net/gtFine/seg/val/'+data.replace('leftImg8bit.png','gtFine_labelIds.png'), cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, dsize=(1024,512), interpolation=cv2.INTER_NEAREST)
    
        sparse_seg_map = np.zeros((512,1024,34))
        for i in range(34):
            sparse_seg_map[seg_map == i, i] = 1
    
        sparse_seg = np.zeros((512,1024,34))
        for i in range(34):
            sparse_seg[seg == i, i] = 1
        m.update_state(sparse_seg, sparse_seg_map)
        miou_val = m.result().numpy()
        val_list.append(miou_val)
    else:
        if depth:
            depth = cv2.imread('E:/cityscapes/DTS_Net/disparity_trainvaltest/disparity/val/'+data.replace('leftImg8bit.png','disparity.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = cv2.resize(depth, dsize=(1024,512), interpolation=cv2.INTER_NEAREST)
            depth[depth > 0] = (depth[depth > 0] - 1) / 256
    
            pred = depth_map
            gt = depth
            ind = np.where(depth == 0)
            gt[gt == 0] = 1
            pred[ind] = 1
            errors.append(compute_errors(gt, pred))
        else:
            seg = cv2.imread('E:/cityscapes/DTS_Net/gtFine/seg/val/'+data.replace('leftImg8bit.png','gtFine_labelIds.png'), cv2.IMREAD_GRAYSCALE)
            seg = cv2.resize(seg, dsize=(1024,512), interpolation=cv2.INTER_NEAREST)
    
            sparse_seg_map = np.zeros((512,1024,34))
            for i in range(34):
                sparse_seg_map[seg_map == i, i] = 1
    
            sparse_seg = np.zeros((512,1024,34))
            for i in range(34):
                sparse_seg[seg == i, i] = 1
            m.update_state(sparse_seg, sparse_seg_map)
            miou_val = m.result().numpy()
            val_list.append(miou_val)
if sim:   
    print('mean IOU: ', sum(val_list)/len(val_list))
    mean_errors = np.array(errors).mean(0)  
    print('mean depth errors: ', mean_errors)
else:
    if depth:
        mean_errors = np.array(errors).mean(0)  
        print('mean depth errors: ', mean_errors)
    else:
        print('mean IOU: ', sum(val_list)/len(val_list))
    