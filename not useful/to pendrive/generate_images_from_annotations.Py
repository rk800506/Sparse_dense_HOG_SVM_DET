import cv2
import os
import xml.etree.ElementTree as ET

image_folder_with_annotation_files = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/bigger_dataset_128_64/neg'
crop_win = [128,64]
folder_to_save_cropped_images = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/bigger_dataset_128_64/neg_128_64'
i = 0
resize_fact = 2
for image in os.listdir(image_folder_with_annotation_files):
    i = i+1
    print(i)
    if image.endswith('.jpg'):
        img_path = os.path.join(image_folder_with_annotation_files,image)

        xml_file_path = os.path.join(image_folder_with_annotation_files, image.replace('.jpg','')+'.xml')
        #print(xml_file_path)
        #print(img_path)
        in_img = cv2.imread(img_path)
        #img_cropped = in_img[:,32:96]
        #cv2.imshow('winname', in_img)
        #print(in_img.shape)
        
        (h,w, c) = in_img.shape
     
        aa = int(w/resize_fact)
        bb = int(h/resize_fact)
        in_img = cv2.resize(in_img,(aa,bb))
        #print(in_img.shape)
        
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        print(image)
        for elem in root.iter():
            if elem.tag == 'xmin':
                xmin = (int(elem.text))/resize_fact
            if elem.tag == 'xmax':
                xmax = (int(elem.text))/resize_fact
            if elem.tag == 'ymin':
                ymin = (int(elem.text))/resize_fact
            if elem.tag == 'ymax':
                ymax = (int(elem.text))/resize_fact
        
        img_cen = [(ymin+ymax)/2,(xmin+xmax)/2]
        a = int(img_cen[0]-crop_win[0]/2)
        b = int(img_cen[0]+crop_win[0]/2)
        c = int(img_cen[1]-crop_win[1]/2)
        d = int(img_cen[1]+crop_win[1]/2)
        #print(img_cen)
        #print(a,b,c,d)
        img_cropped = in_img[a:b,c:d]
        
        print(img_cropped.shape)
        if img_cropped.shape == (128,64,3):
            f_crp_name = os.path.join(folder_to_save_cropped_images, image)
            cv2.imwrite(f_crp_name, img_cropped)    
        