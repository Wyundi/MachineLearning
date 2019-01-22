#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import numpy as np

img = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
print(img)

img_h = 3
img_w = 3

kernal = np.array([-1,0,1,1]).reshape(2,2)

kernal_h = 2
kernal_w = 2

n_h = img_h - kernal_h + 1
n_w = img_w - kernal_w + 1

img2col_h = n_h * n_w
img2col_w = kernal_h * kernal_w

b = np.zeros((img2col_h, img2col_w))
print(b.shape)
print(img[0:2, 0:2].reshape(1,4))

# im2col
for i in range(n_h):
    for j in range(n_w):
        b[i*n_h+j] = img[i:i+kernal_h, j:j+kernal_w].reshape(1, 4)

kernal = kernal.reshape(4,1)

img_conv = np.dot(b, kernal).reshape(2,2)
print(img_conv)

