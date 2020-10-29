import cv2

import tensorflow.compat.v1 as tf
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import random
import os
#
# print(os.getcwd())
# path_dir = 'C:/Users/TREE/Desktop/HandPose-master/Poses/OK/OK_1'
# file_list = os.listdir(path_dir)
# print(file_list)

tf.disable_v2_behavior()


print(os.listdir("Poses/"))

poses = os.listdir('Poses/')


# print(os.listdir("Poses/"))

# poses = os.listdir('Poses/')

for pose in poses:
    print(">> Working on pose : " + pose)
    subdirs = os.listdir('Poses/' + pose + '/')
    for subdir in subdirs:
        files = os.listdir('Poses/' + pose + '/' + subdir + '/')
        print(">> Working on examples : " + subdir)
        for file in files:
            if(file.endswith(".png")):
                exist = file.find('clock') + file.find('counter') + file.find('horizon')
                if exist == -3:
                    print(file)
                    path = 'Poses/' + pose + '/' + subdir + '/'  + file
                    path_clock = 'Poses/' + pose + '/' + subdir + '/' + 'clock' + file
                    path_counter = 'Poses/' + pose + '/' + subdir + '/' + 'counter' + file
                    path_180 = 'Poses/' + pose + '/' + subdir + '/' + 'horizon' + file

                    # Read image
                    im = cv2.imread(path)

                    # im_90_clock = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                    # im_90_counter = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    im_180 = cv2.flip(im, 1)

                    # cv2.imwrite(path_clock,im_90_clock)
                    # cv2.imwrite(path_counter, im_90_counter)
                    cv2.imwrite(path_180, im_180)
