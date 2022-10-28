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
model_path = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/model_name.npy'
model = joblib.load(model_path)
print(model)

# define the sliding window:
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
            
            
def get_detection_boxes_image_pyramid(image, downscale, window_size, stride_slide_win, scale=0):
    [winH, winW] = window_size
    count_hog = 0
    for resized in pyramid_gaussian(image, downscale=downscale, max_layer = 4): 
        for (x,y,window) in sliding_window(resized, stepSize=stride_slide_win, windowSize=window_size):
            count_hog += 1
            #print(count_hog)
            if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met 
                continue

            #st_time = time.time()
            fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
            #print(time.time()-st_time)

            fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog
            pred = model.predict(fds) 
            
            if pred == 1:
                if model.decision_function(fds) > 0.5:
                    #print(int(x * (downscale**scale)), int(y * (downscale**scale)))
                    #print("Detection:: Location -> ({}, {})".format(x, y))
                    #print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                    int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                        int(windowSize[1]*(downscale**scale))))
                    return detections

        scale+=1
    print(count_hog)
    return detections



orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
threshold = .3
stride_slide_win = [20,20]
windowSize = [128,64]
downscale = 1.5
detections = []
scale = 0

#### check gaussian pyramis images ##
from matplotlib import pyplot as plt
%matplotlib inline
import cv2 
import time

inp_img_path = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/294.jpg'
img_folder = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/temp3'
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
    print('detection time: ', (time.time()-st))
    #print(detections)
    

