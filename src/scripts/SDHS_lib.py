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


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])



def get_detection_boxes_image_pyramid(image, detect_model, downscale, window_size, stride_slide_win, scale=0, \
    layers = 4, CS_th = 0.2, image_start = [0,0], orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2)):
    [winH, winW] = window_size
    count_hog = 0
    detections = []
    hog_count_cum = 0
    model = detect_model

    for resized in pyramid_gaussian(image, downscale=downscale, max_layer = layers): 
        for (x,y,window) in sliding_window(resized, stepSize=stride_slide_win, windowSize=window_size):
            count_hog += 1
            if window.shape[0] != winW or window.shape[1] !=winH: # ensure the sliding window has met
                #print('here')
                continue
            fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
            fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog
            pred = model.predict(fds) 

            if pred == 1:
                if model.decision_function(fds) > CS_th:
                    detections.append((int(x * (downscale**scale))+image_start[0], int(y * (downscale**scale))+image_start[1], model.decision_function(fds),
                                    int(window_size[0]*(downscale**scale)), # create a list of all the predictions found
                                        int(window_size[1]*(downscale**scale)), scale))
                    #return detections

        scale+=1
        hog_count_cum += count_hog
        count_hog = 0
    return detections, hog_count_cum



def dense_detect(orig_image, detect_model, detections):
    dense_detections = []
    total_dense_hog_c = 0
    for item in detections:
        #print(item)
        image = orig_image[item[1]:item[1]+item[4], item[0]:item[0]+item[3]]
        #print(image.shape)
        #plt.figure()
        ##plt.imshow(image)
        #plt.show()
        detections,hog_count = get_detection_boxes_image_pyramid(image, detect_model, downscale=1.1, \
            stride_slide_win= [8, 8], window_size=[64, 128], layers = 0,\
                CS_th =0.3, image_start = [item[0],item[1]])
        #
        dense_detections = dense_detections+detections
        total_dense_hog_c += hog_count
        #print(item, dense_detections)
    return dense_detections, total_dense_hog_c




def sparse_detect(img, detect_model,  sparse_slide_wind):
    #detections = []
    #[w,h] = img.shape
    #st = time.time()
    detections,hog_count = get_detection_boxes_image_pyramid(img, detect_model, downscale=1.1, \
        stride_slide_win= sparse_slide_wind, window_size=[64, 128], layers = 5,\
                                                  CS_th = 0.02)
    #print(stride_slide_win)
    return detections, hog_count