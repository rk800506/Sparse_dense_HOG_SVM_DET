#!env /usr/bin/python3

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

### load trained model
model_path = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/trained_models/hog_models_128_64/hogsvm_pedes_9bins.npy'
model = joblib.load(model_path)
print(model)

# define the sliding window:
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
            
            
def get_detection_boxes_image_pyramid(image, downscale, window_size, stride_slide_win, scale=0, layers = 4, CS_th = 0.2, image_start = [0,0]):
    [winH, winW] = window_size
    count_hog = 0
    detections = []
    hog_count_cum = 0
    #print(window_size, [winH, winW])
    for resized in pyramid_gaussian(image, downscale=downscale, max_layer = layers): 
        #print('resized shape:', resized.shape)
        for (x,y,window) in sliding_window(resized, stepSize=stride_slide_win, windowSize=window_size):
            count_hog += 1
            #print(x,y,window.shape)
            if window.shape[0] != winW or window.shape[1] !=winH: # ensure the sliding window has met
                #print('here')
                continue

            #st_time = time.time()
            fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
            #print('here too')
            #print(time.time()-st_time)

            fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog
            pred = model.predict(fds) 
            #print('scale = ', scale)
            if pred == 1:
                if model.decision_function(fds) > CS_th:
                    #print('detected at sclae:' , scale)
                    #print(int(x * (downscale**scale)), int(y * (downscale**scale)))
                    #print("Detection:: Location -> ({}, {})".format(x, y))
                    #print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                    detections.append((int(x * (downscale**scale))+image_start[0], int(y * (downscale**scale))+image_start[1], model.decision_function(fds),
                                    int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                        int(windowSize[1]*(downscale**scale)), scale))
                    #return detections

        scale+=1
        hog_count_cum += count_hog
        #print('hog count:', count_hog)
        count_hog = 0
        #print('cale:', scale)
    return detections




orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
threshold = .8
stride_slide_win = [20,20]
windowSize = [64,128]
downscale = 1.5
detections = []
scale = 0

#### check gaussian pyramis images ##
from matplotlib import pyplot as plt
#%matplotlib inline
import cv2 
import time

inp_img_path = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/294.jpg'
img_folder = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/raw_images/temp3'
count = 0

for img in os.listdir(img_folder):
    count += 1
    img_path = os.path.join(img_folder, img)
    #print(img_path)
    image = cv2.imread(img_path, flags=0)
    [w,h] = image.shape
    image = cv2.resize(image, (h//2,w//2))
    st = time.time()
    detections = get_detection_boxes_image_pyramid(image, downscale=downscale, stride_slide_win= stride_slide_win, window_size=windowSize)
    print(detections)
    print('detection time: ', (time.time()-st))
    #print(detections)

    for item in detections:
        image = cv2.rectangle(image, (item[0],item[1]), (item[0]+item[3], item[1]+item[4]), (255,0,0), 1)
    cv2.imshow('frane', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
