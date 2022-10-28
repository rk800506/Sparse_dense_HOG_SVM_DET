# Importing the necessary modules:

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
#from sklearn.externals
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)


# define path to images:

pos_im_path = r"/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/CPim5" # This is the path of our positive input dataset
#pos_im_path = r"Insert\path\for\positive_images\here" # This is the path of our positive input dataset
# define the same for negatives
#neg_im_path= r"Insert\path\for\negative_images\here"
neg_im_path= r"/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/CNim5"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)
print(num_pos_samples) # prints the number value of the no.of samples in positive dataset
print(num_neg_samples)
data= []
labels = []

# compute HOG features and label them:

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
    #print(fd.shape)
    data.append(fd)
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
    data.append(fd)
    labels.append(0)
    #print(labels)
# encode the labels, converting them from strings to integers
print(size(fd))
print(size(data))
print(size(labels))
le = LabelEncoder()
labels = le.fit_transform(labels) 
print(labels)


#%%
# Partitioning the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print(" Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.20, random_state=42)
#%% Train the linear SVM
print(" Training Linear SVM classifier...")
#model = LinearSVC()
model  = LinearSVC(max_iter=10000)
model.fit(trainData, trainLabels)
#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))


# Save the model:
#%% Save the Model
joblib.dump(model, 'model_name.npy')


# %%