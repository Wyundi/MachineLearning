#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

# Multivariate Linear Regression

import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time

class LinearRegression():
    def __init__(self, X, y):
        self.m = X.shape[0]                     # number of examples
        self.n = X.shape[1]                     # number of features
        self.x_extra = np.ones([self.m, 1])
        self.data_X = np.column_stack((self.x_extra, X))
        self.data_y = y
        self.__theta = np.random.rand(self.m, 1)
        self.__alpha = 0.01
        self.h = np.random.rand(self.m, 1)
        self.J = 0
        self.J_iteration = np.zeros([1,1])
        self.D = np.random.rand(self.n, 1)
        self.temp = np.random.rand(self.m, 1)

    def modifyParam(alpha):
        self.__alpha = alpha

    def Hypothesis(self):
        for i in range(self.m):
            for j in range(self.n):
                self.h[i] += self.__theta.T[0][j] * self.data_X[i][j]

    def CostFunction(self):
        self.Hypothesis()
        for i in range(self.n):
            self.J += 1/(2*self.m) * np.power((self.h[i] - self.data_y[0][i]), 2)

    def Derivative(self):
        self.Hypothesis()
        for i in range(self.n):
            for j in range(self.m):
                self.D[i] += 1/self.m * (self.h[j] - self.data_y[0][j]) * self.data_X[j][i]

    def UpdateParam(self):
        self.Derivative()
        self.temp = self.__theta - self.__alpha * self.D.T
        self.__theta = self.temp
    
class Visualize():
    # 数据可视化
    def plot(self, x, y):
        plt.plot(x, y, color = 'blue', label = 'plot')
    
    def scatter(self, x, y):
        plt.scatter(x, y, color = 'green', label = 'point')
    
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

def main():
    # diabetes datasets
    diabetes = datasets.load_diabetes()
    data = diabetes.data
    target = np.array([diabetes.target])

    # print(data.shape)             ->(442, 10)
    # print(target.shape)           ->(1, 442))

    x = data
    y = target

    # 将数据分为训练集和测试集
    x_train, x_test = x[:-20], x[-20:]
    y_train, y_test = y[:,:-20], y[:,-20:]

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    LR = LinearRegression(x_train, y_train)
    PLT = Visualize()

    LR.Hypothesis()
    print(LR.h.shape)                           # ->(422, 1)

    LR.CostFunction()
    print(LR.J)

    for i in range(1000):
        LR.Derivative()

    PLT.plot(x_train, LR.h)

    PLT.scatter(x_train, y_train.T)
    # PLT.axis()
    PLT.cross()
    PLT.show()

######
main()