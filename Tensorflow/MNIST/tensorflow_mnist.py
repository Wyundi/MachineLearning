#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

# TensorFLow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Other libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:\t", tf.__version__)

# datasets
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0


# 检查数据
'''
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()
'''

# 构建模型

# 设置层
# Flatten: 扁平化像素, input = （28, 28） -> (784, 1)
# Dense: 全连接层, (128, ReLU), (10, Softmax)
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

# 编译模型
# 优化器 - 损失函数 - 指标(准确率)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# epochs: 训练次数
model.fit(train_images, train_labels, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# 预测
predictions = model.predict(test_images)

print("prediction:", np.argmax(predictions[0]), "\t", "labels:", test_labels[0])