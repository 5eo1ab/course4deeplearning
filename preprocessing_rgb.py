# -*- coding: utf-8 -*-
# Hanbin Seo (github.com/5eo1ab)
# 2017.10.14.
# preprocessing_rgb.py
# image data => RGB numeric data
# raw data URL: https://www.kaggle.com/c/dogs-vs-cats
#####################################

import os
# (tf-v1.4) seo1ab@seo1ab-All-Series:~$ pip install opencv-python
import cv2
import numpy as np

print(os.getcwd())  # /home/seo1ab/PycharmProjects/CourseWork
os.chdir('./assignment_CNN/train')

f_nm_list = os.listdir()
print(len(f_nm_list))

""" # test code
re_img1 = cv2.imread('cat.1.jpg')
print(re_img1.shape)  # (280, 300, 3)
print(re_img1.size)  # 252000  = 280*300*3
print(type(re_img1))  # <class 'numpy.ndarray'>

b, g, r = cv2.split(re_img1)
print(b.shape)  # (280, 300)
print(b.size)  # 84000

rs_img1 = cv2.resize(re_img1, (250,250))
print(rs_img1.shape)  # (250, 250, 3)
print(rs_img1.size)  # 187500
cv2.imwrite('../resize.cat.1.jpg', rs_img1)

print(rs_img1[-1].shape) # (250, 3)
arr_img1 = np.reshape(rs_img1, (rs_img1.size,))
print(arr_img1.shape)  # (187500,)

rc_img1 = np.reshape(arr_img1, (250,250,3))
print(rc_img1.shape)  # (250, 250, 3)
cv2.imwrite('../recover.cat.1.jpg', rc_img1)

re_list = list()
for i in range(10):
    re_img = cv2.imread('cat.{}.jpg'.format(i))
    rs_img = cv2.resize(re_img, (128,128))
    arr_img = np.reshape(rs_img, (rs_img.size,))
    re_list.append(arr_img)
print(np.stack(re_list).shape) # (10, 49152)
print(np.stack(re_list))

arr_img.shape
np.append(arr_img, 0).shape
"""

def convert_RGB_array(fname, unit_size=100):
    object_nm = f_nm.split('.')[0]
    label = 1 if object_nm != 'cat' else 0  # (1 = dog, 0 = cat)
    re_img = cv2.imread(f_nm)
    rs_img = cv2.resize(re_img, (unit_size, unit_size))
    arr_img = np.reshape(rs_img, (rs_img.size,))
    lb_img = np.append(arr_img, label)
    return lb_img

list_rgb = list()
for i, f_nm in enumerate(f_nm_list):
    arr_rgb = convert_RGB_array(f_nm, unit_size=128)
    list_rgb.append(arr_rgb)
    if i%100==0: print("{}/{}".format(i, len(f_nm_list)))
np_data = np.stack(list_rgb)
print(np_data.shape)  # (25000, 49153)


print(os.getcwd())  # /home/seo1ab/PycharmProjects/CourseWork/assignment_CNN/train
os.chdir('../')
print(os.getcwd())  # /home/seo1ab/PycharmProjects/CourseWork/assignment_CNN

np.save('DataRGB.npy', np_data)
