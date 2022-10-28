import cv2
import os

#raw_img_folder = "/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/raw_img2"
src_fold = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/train/neg'
#raw_img_folder = '/home/dtu-project2/Downloads/raw_img'
#folder_to_save_img = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/temp2'
dest_fol = '/media/dtu-project2/2GB_HDD/Detection_HOG_SVM/dataset/bigger_dataset_128_64/train/neg_flipped'
#folder_to_save_img = '/home/dtu-project2/Downloads/cropped_img'
win_size = [128, 128]
stride = [256, 256] # sttride vertical and stride hprizontal respectively
i = 0
#global total_imgs

## code to crop and generate flipped versions of input image ###
def flip_image(img, orientation):
    if orientation == 'h':
        image = cv2.flip(img, 1)
    else:
        image = cv2.flip(img, 0)
    return image


def main():
    temp = 0
    for file in os.listdir(src_fold):
        img_path = os.path.join(src_fold, file)
        #print(img_path)
        in_img = cv2.imread(img_path)
        #cv2.imshow('winname', in_img)
        [h, w, c] = in_img.shape
        flip_h = flip_image(in_img, 'h')
        flip_v = flip_image(in_img, 'v')
        #h_img_p = img_path.replace('.jpg','_h.jpg')
        #v_img_p = img_path.replace('.jpg','_v.jpg')
        cv2.imwrite(h_img_p, flip_h)
        cv2.imwrite(v_img_p, flip_v)
        #print(h,w,c)
        '''
        print(h, w)
        if h > 600 and w > 800:
            in_img = cv2.resize(in_img, (640, 360))
            print(in_img.shape)
            #cv2.imwrite('resized_mig.jpg', in_img)
        print(h, w, c)
        
        if h < win_size[0] or w < win_size[1]:
            print('image dim is less than the crop window')
        else:
            remh = (h - win_size[0])%stride[0]
            remw = (w - win_size[1])%stride[1]
            #print(remh, remw, h, w)
            #print(in_img.shape)
            im_res = cv2.resize(in_img, (w-remw, h-remh))
            #print(im_res.shape)
            [hh, ww, cc] = im_res.shape
    
            num_img_vert = (hh-win_size[0])/stride[0] +1
            num_img_horz = (ww - win_size[1])/stride[1]+1
    
            num_imgs = num_img_vert*num_img_horz
            #print(hh,(hh-win_size[0])/stride[0] + 1, ww, (ww - win_size[1])/stride[1]+1)
            print('num_imgs: ', num_imgs)
            temp = temp+ num_imgs
            for i in range(int(num_img_vert)):
                for j in range(int(num_img_horz)):
                    filename = str(i)+'_'+str(j)+'_'+file
                    filepath = os.path.join(folder_to_save_img, filename)
                    print(filepath)
                    #crop_top_left_loc = [0+stride[0]*i, 0+stride[1]*j]

    
                    # window
                    top_left = [j*stride[1], i*stride[0]]
                    #down_left = i*stride[0]+j*stride[1]+win_size[0]
    
                    #top_right = i*stride[0]+j*stride[1] +win_size[1]
                    #down_right = i*stride[0]+j*stride[1] + win_size[1]+ win_size[0]
    
                    #print(top_left)
    
                    cropper_img = im_res[top_left[1]:top_left[1]+win_size[0],top_left[0]:top_left[0]+win_size[1]]
                    #print(cropper_img.shape)
                    cv2.imwrite(filepath, cropper_img)
                    #print()
        print(temp)
        '''
    print("task complere")

if __name__ == '__main__':
    
    main()
