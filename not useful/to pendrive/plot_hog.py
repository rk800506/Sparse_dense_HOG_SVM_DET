from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt
 
man = imread('CPim5/90.jpg', as_gray=True)
# https://en.wikipedia.org/wiki/German_Shepherd#/media/File:Kim_at_14_weeks.jpg

man_hog, man_hog_img = hog(
    man, pixels_per_cell=(8,8), 
    cells_per_block=(2, 2), 
    orientations=3, 
    visualize=True, 
    block_norm='L2-Hys')

print(man_hog.shape)
 
fig, ax = plt.subplots(1,2)

# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
    for a in ax]
 
ax[0].imshow(man, cmap='gray')
ax[0].set_title('man')
ax[1].imshow(man_hog_img, cmap='gray')
ax[1].set_title('hog')
plt.show()