# Importing the necessary modules:
#import joblib
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
from skimage.feature import hog


# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# define path to images:
pos_im_path = r"/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/pos" # This is the path of our positive input dataset
neg_im_path = r"/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/neg"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)
print(num_pos_samples) # prints the number value of the no.of samples in positive dataset
print(num_neg_samples)
data= []
labels = []

# prepare data in form of matrix and each row is HOG feature of a single image
# compute HOG features and label them:

data_mat = np.ndarray((1,3780))


for file in pos_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one
    filename = os.path.join(pos_im_path, file)
    #print(file)
    img = Image.open(filename)#+ '\\/home/ubuntu18/Downloads/Object-detection-via-HOG-SVM-master/positive') # open the file 
    #img = Image.open(pos_im_path + '\\' + file) # open the file
    
    #img = img.resize((240,240))
    #img = img.resize((64,128))
    gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
    # calculate HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    fd = np.expand_dims(fd, axis=0)

    print(fd.shape)
    data_mat = np.vstack([data_mat, fd])
    #fd = np.transpose(fd)
    #print(fd.shape)

    #print(size(data))
    labels.append(1)
    #print(labels)

print(size(data))
# Same for the negative images
for file in neg_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one
    filename = os.path.join(neg_im_path, file)
    #print(file)
    img = Image.open(filename)
    #img=Image.open(neg_im_path + '\\' + file)
    #img = img.resize((240,240))
    #img = img.resize((64,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    fd = np.expand_dims(fd, axis=0)

    data_mat = np.vstack([data_mat, fd])
    labels.append(0)
    #print(labels)

# create SVM object
svm = cv2.ml.SVM_create()

print(data_mat.shape)
labels_mat = np.array((labels))
print(svm)


