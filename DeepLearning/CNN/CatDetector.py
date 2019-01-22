#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.model_selection import train_test_split
import cv2 as cv

# 显示图片  (imshow + waitKey)

'''
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''
def imshow(img):
    cv.imshow('Cat', img)
    k = cv.waitKey(0) & 0xFF
    if k == ord('q'):
        cv.destroyWindow('Cat')           # 销毁单个图片
    elif k == ord('s'):
        cv.imwrite('cat.1.gray.jpg', img)

    cv.destroyAllWindows()      # 销毁全部图片

def CNN(img):
    print(img.shape)

def main():
    # 读取图片
    img = cv.imread('cat.jpg', 1)
    CNN(img)

main()