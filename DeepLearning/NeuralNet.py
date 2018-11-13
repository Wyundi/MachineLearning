#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification

class Visualize():
    # 数据可视化
    def plot(self, x, y):
        plt.plot(x, y, color = 'blue', label = 'plot')
    
    def scatter(self, x, y, c):
        plt.scatter(x, y, color = c, label = 'point')

    def plot_surface(self, x, y, z):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z, rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

    def contourf(self, x, y, z):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
    
    def axis(self, x0, x1, y0, y1):
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)

    def cross(self):
        plt.ax = plt.gca()
        plt.ax.spines['right'].set_color('none')
        plt.ax.spines['top'].set_color('none')
        plt.ax.xaxis.set_ticks_position('bottom')
        plt.ax.spines['bottom'].set_position(('data', 0))
        plt.ax.yaxis.set_ticks_position('left')
        plt.ax.spines['left'].set_position(('data', 0))
    def show(self):
        plt.show()

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.where(z<0, 0, z)

def NeuralNet(X, y):
    m = X.shape[0]
    n = X.shape[1]
    Layer = np.array([3, 1])

    W1 = np.random.rand(n, Layer[0])
    b1 = np.random.rand(m, 1)
    W2 = np.random.rand(Layer[0], Layer[1])
    b2 = np.random.rand(m, 1)

    Z1 = np.zeros([m, Layer[0]])
    A1 = np.zeros([m, Layer[0]])
    Z2 = np.zeros([m, Layer[1]])
    A2 = np.zeros([m, Layer[1]])

    Loss = np.zeros([m, 1])
    J = 0

    alpha = 0.01
    
    while(True):
        # 正向传播
        Z1 = np.dot(X, W1) + b1
        A1 = ReLU(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = Sigmoid(Z2)

        # 反向传播
        Loss = - y*(np.log(A2)) - (1-y)*(np.log(1-A2))

        J0 = J
        J = 1/m * np.sum(Loss)
        # print(J)
        J_dv = np.fabs(J-J0)
        
        # dx 和 x 的维度相同
        dZ2 = (A2 - y) * (A2*(1-A2))
        dW2 = 1/m * np.dot(A1.T, dZ2)
        db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)

        dZ1 = np.dot(dZ2, W2.T) * np.where(Z1<0, 0, 1)
        dW1 = 1/m * np.dot(X.T, dZ1)
        db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

        #参数更新
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1

        if J_dv <= 0.000001:
            break
    
    return W1, W2, b1, b2

def main():
    # x, y = make_moons(n_samples=20000,noise=0.1)
    x, y = make_circles(n_samples=1000,factor = 0.7,noise=0.05)

    y = y.reshape([x.shape[0], 1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

    print(x.shape, y.shape)

    W1, W2, b1, b2 = NeuralNet(x_train, y_train)

    Z1 = np.dot(x_test, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Sigmoid(Z2)

    m = 0
    for i in range(x_test.shape[0]):
        if A2[i] >= 0.5:
            A2[i] = 1
        else:
            A2[i] = 0
        if A2[i] != y_test[i]:
            m = m + 1
        print(A2[i], '  ', y_test[i])
    print(m)

    PLT = Visualize()

    for i in range(y.shape[0]):
        if y[i] == 0:
            PLT.scatter(x[i, 0], x[i, 1], 'green')
        else:
            PLT.scatter(x[i, 0], x[i, 1], 'red')

    PLT.cross()
    PLT.show()

main()