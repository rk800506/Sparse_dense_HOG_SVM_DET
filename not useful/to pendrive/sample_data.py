import os
import sys
import shutil
main_dir = '/media/dtu-project2/2GB_HDD/tensorflow2/workspace/training_demo/images/train'
number_images_needed = 3000
save_folder = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/bigger_dataset/pos'


dir_list  = os.listdir(main_dir)

i = 0
for img in dir_list:
    if img.endswith('.jpg'):
        img_path = os.path.join(main_dir, img)
        #print(img)
        xml_path = os.path.join(main_dir, img.replace('.jpg', '')+'.xml')
        i += 1
        print(i)
        if i <= number_images_needed:
            shutil.copy(src=img_path, dst=save_folder)
            shutil.copy(src=xml_path, dst=save_folder)