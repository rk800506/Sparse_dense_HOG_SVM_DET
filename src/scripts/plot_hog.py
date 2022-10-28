from skimage.feature import hog
from skimage.io import imread, imsave
from skimage.transform import rescale
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
 
man = imread('/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/histogram plots/3434.jpg', as_gray=True)
# https://en.wikipedia.org/wiki/German_Shepherd#/media/File:Kim_at_14_weeks.jpg
#img_fold = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/histogram plots/rgb_img'
#hog_feat_fold = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/histogram plots/hogs'


man_hog, man_hog_img = hog(
    man, pixels_per_cell=(8,8), 
    cells_per_block=(2, 2), 
    orientations=3, 
    visualize=True, 
    block_norm='L2-Hys')

print(man_hog_img.shape)

fig, ax = plt.subplots(1,2)

# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
    for a in ax]
 
ax[0].imshow(man, cmap='gray')
ax[0].set_title('man')
ax[1].imshow(man_hog_img, cmap='gray')
ax[1].set_title('hog')
plt.show()
#plt.imsave('hog_feat.jpg', man_hog_img)

'''
for img in os.listdir(img_fold):
    img_path = os.path.join(img_fold, img)
    image = imread(img_path, as_gray=True)
    #print(image.shape)
    fd, man_hog_img = hog(
        man, pixels_per_cell=(16,16), 
        cells_per_block=(2, 2), 
        orientations=9, 
        visualize=True, 
        block_norm='L2-Hys')
    print(man_hog_img)
    hog_path = os.path.join(hog_feat_fold, img.replace('.jpg','')+'hog.jpg')
    #man_hog_img = cv2.bitwise_not(man_hog_img)
    cv2.imwrite(hog_path, man_hog_img)
    
'''
