#!/usr/bin/env python

from SDHS_lib import*

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
from PIL import Image

#initialize deector parameters
orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
threshold = .3
stride_slide_win = [64,32]
#stride_slide_win = [10,10]
stride_slide_win_dense = [10,10]
windowSize = [64,128]
downscale = 1.5
detections = []
scale = 0
max_layers = 4
cs_thresh_sparse = 0.02
cs_thresh_dense = 0.3

############################################
# read image folder or video
img_dir = "/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/raw_images/raw_img2"
dir_list = os.listdir(img_dir)
dir_list.sort()


if dir_list == []:
    print('img dir empty')

## load detection model: the trained on UaV123 dataset
model_path = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/trained_models/hog_models_128_64/hogsvm_pedes_9bins.npy'
model = joblib.load(model_path)



for img_name in dir_list:
    img_path = os.path.join(img_dir, img_name)
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # sparse dense hog svm detection
    detecti, hog_count_sparse = sparse_detect(img, model,sparse_slide_wind=stride_slide_win)
    dense_detections, hog_count_dense = dense_detect(img, model, detecti)

    cv2.imshow("frame", img)
    cv2.waitKey(100)
cv2.destroyAllWindows()
