# Sparse_dense_HOG_SVM_DET
Repository hosting code for sparse dense sampling based 2 stage HOG-SVM object detector

## About this repository
This repository hsots the code for the work presented in **A Sparse-Dense HOG Window Sampling Technique for Fast Pedestrian Detection in Aerial Images**.
A Two stage sparse-dense sampling based technique has been presented that fastens up the standard object detection using HOG-SVM classifier. First stage (spase sampling) filters out the most relevant regions in an image quickly, the second stage (dense sampling) only checks the proposed regions for any object.

Standard aerial image dataset **UAV123** has been used to validate the technique. The dataset (modified to train the classifier) used in this work can be downloaded from https://drive.google.com/file/d/1RAgHgdAeXSSNMbc2kPp_wzUvwD2HKS8G/view?usp=sharing.

#### Note:
Further instructions on how to use the code will be added soon.

#### If you use the work, cite the following paper:

Kumar, R., Deb, A.K. (2022). A Sparse-Dense HOG Window Sampling Technique for Fast Pedestrian Detection in Aerial Images. In: Mekhilef, S., Shaw, R.N., Siano, P. (eds) Innovations in Electrical and Electronic Engineering. ICEEE 2022. Lecture Notes in Electrical Engineering, vol 893. Springer, Singapore. https://doi.org/10.1007/978-981-19-1742-4_37
