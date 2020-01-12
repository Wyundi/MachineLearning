#!/usr/bin/env python
# -*- coding:utf-8 -*-
# https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn

from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow
import tensorflow as tf
import numpy as np
import os

path = "/home/wyundi/Project/Git/MachineLearning/Tensorflow/Tutorial"

# 获取数据集 
x_train = np.load(path + "/mnist_npz/x_train.npy")
y_train = np.load(path + "/mnist_npz/y_train.npy")
x_test = np.load(path + "/mnist_npz/x_test.npy")
y_test = np.load(path + "/mnist_npz/y_test.npy")

x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建神经网络
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练并验证模型
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)