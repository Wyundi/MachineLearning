#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualize():
    # 数据可视化
    def plot(self, x, y):
        plt.plot(x, y, color = 'blue', label = 'plot')
    
    def scatter_green(self, x, y):
        plt.scatter(x, y, color = 'green', label = 'point')

    def scatter_red(self, x, y):
        plt.scatter(x, y, color = 'red', label = 'point')

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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    x1 = np.array([1,1,1,2,2,3,3,4,4,5,5,5])
    x2 = np.array([1,2,3,1,2,1,3,3,2,3,2,1])
    y = np.array([[0,0,0,0,0,0,1,1,1,1,1,1]]).T
    x_e = np.ones([12, 1])
    X = np.column_stack((x_e, x1, x2))
    theta = np.zeros([3, 1])
    alpha = 0.01
    J = 0
    D = np.zeros([3,1])
    
    while(True):
        h = sigmoid(np.dot(X, theta))
        Loss = - y*(np.log(h)) - (1-y)*(np.log(1-h))
        J0 = J
        J = 1/11 * np.sum(Loss)
        print(J)
        J_dv = np.fabs(J-J0)

        for i in range(3):
            D[i] = 1/12 * np.sum((h - y) * X[:, i].reshape(12, 1))

        theta = theta - alpha * D

        if J_dv <= 0.00001:
            break

    img = np.dot(X, theta)
    print(theta)
    PLT = Visualize()
    
    for i in range(y.shape[0]):
        if y[i] == 0:
            PLT.scatter_green(x1[i], x2[i])
        else:
            PLT.scatter_red(x1[i], x2[i])
    
    x1_t = np.array([1,2,3,4,5])
    x2_t = - (theta[0] + theta[1] * x1_t) / theta[2]

    PLT.plot(x1_t, x2_t)

    # PLT.cross()
    PLT.show()


main()