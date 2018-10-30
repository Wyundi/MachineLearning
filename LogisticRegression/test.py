#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import numpy as np
import matplotlib.pyplot as plt

class Visualize():
    # 数据可视化
    def plot(self, x, y):
        plt.plot(x, y, color = 'blue', label = 'plot')
    
    def scatter(self, x, y):
        plt.scatter(x, y, color = 'green', label = 'point')

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
    x = np.arange(0,11).reshape(11,1)
    y = np.array([[0,0,0,0,1,0,0,1,1,1,1]]).T
    x_e = np.ones([11, 1])
    X = np.column_stack((x_e, x))
    theta = np.zeros([2, 1])
    alpha = 0.01
    J = 0

    while(True):
        h = sigmoid(np.dot(X, theta))
        Cost = - y*(np.log(h)) - (1-y)*(np.log(1-h))
        J0 = J
        J = 1/11 * np.sum(Cost)
        print(J)
        J_dv = np.fabs(J-J0)

        D = np.zeros([2,1])

        for i in range(2):
            D[i] = 1/11 * np.sum((h - y) * X[:, i].reshape(11, 1))

        theta = theta - alpha * D

        if J_dv <= 0.000000000000001:
            break

    PLT = Visualize()
    PLT.scatter(x, y)
    PLT.plot(x, h)
    PLT.cross()
    PLT.show()

main()