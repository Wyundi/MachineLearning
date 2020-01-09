#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import numpy as np
import skimage.data
import 

img = skimage.data.chelsea()
img = skimage.color.rgb2gray(img)

l1_filter = numpy.zeros((2, 3, 3))
