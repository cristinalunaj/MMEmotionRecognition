import numpy as np
import cv2
import os
from PIL import Image

def resize(img, new_size=(48,48)):
    return img.resize(new_size)


def convert2grayscale(img):
    return img.convert('LA')


def create_dilation(img, kernel_size=2, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8) *0.5
    dilation = cv2.dilate(img, kernel, iterations=iterations)
    return dilation


def soften_img(img, threshold=38):
    img[img >=threshold] = 255 #white
    img[img<threshold] = 127 #gray
    return img


def convert2grayAndResize(img_subfolder, in_path_imgs,out_path_imgs, newSize):
    #if(os.path.exists(os.path.join(out_path_imgs, img_subfolder))):return
    in_path_subf = os.path.join(in_path_imgs, img_subfolder)
    for img_name in os.listdir(in_path_subf):
        try:
            if (os.path.exists(os.path.join(out_path_imgs, img_subfolder, img_name.rsplit(".")[0] + ".png"))): continue
            img = Image.open(os.path.join(in_path_imgs, img_subfolder, img_name))
            grayScale = convert2grayscale(img)
            imgResized = resize(grayScale, new_size=newSize)
            # plt.imshow(imgResized)
            # plt.show()
            # Save images:
            os.makedirs(os.path.join(out_path_imgs, img_subfolder), exist_ok=True)
            imgResized.save(os.path.join(out_path_imgs, img_subfolder, img_name.rsplit(".")[0] + ".png"))
        except:
            print("Problems with IMAGE: ", os.path.join(img_subfolder, img_name))