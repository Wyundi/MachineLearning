#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time

class LinearRegression():
    def __init__(self, x, y):
        self.data_x = x
        self.data_y = y
        self.__theta0 = 0
        self.__theta1 = 0
        self.__alpha = 0.01
        self.h = np.random.rand(x.shape[0],x.shape[1])
        self.J = 0
        self.D0 = 0
        self.D1 = 0
        self.temp0 = 0
        self.temp1 = 0

    def getParam(self):
        print(self.__theta0, self.__theta1)

    def modifyParam(self, theta0, theta1, alpha):
        self.__theta0 = theta0
        self.__theta1 = theta1
        self.__alpha = alpha

    def Hypothesis(self):
        for i in range(self.data_x.shape[0]):
            self.h[i] = self.__theta0 + self.__theta1 * self.data_x[i]

    def CostFunction(self):
        for i in range(self.data_x.shape[0]):
            self.J += 1/2*self.data_x.shape[0] * np.power(((self.__theta0 + self.__theta1 * self.data_x[i]) - self.data_y[0][i]), 2)
    
    def Derivative(self):
        for i in range(self.data_x.shape[0]):
            self.D0 += 1/self.data_x.shape[0] * ((self.__theta0 + self.__theta1 * self.data_x[i]) - self.data_y[0][i])
            self.D1 += 1/self.data_x.shape[0] * ((self.__theta0 + self.__theta1 * self.data_x[i]) - self.data_y[0][i]) * self.data_x[i]

    def UpdateParam(self):
        self.Derivative()
        self.temp0 = self.__theta0 - self.__alpha * self.D0
        self.temp1 = self.__theta0 - self.__alpha * self.D1
        self.__theta0 = self.temp0
        self.__theta1 = self.temp1

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

    # print(data.shape)             ->(442,10)
    # print(target.shape)           ->(442,)

    # Only one feature  单一变量
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # print(diabetes_X.shape)       ->(442,1)

    x = diabetes_X
    y = target

    # 将数据分为训练集和测试集
    x_train, x_test = x[:-20], x[-20:]
    y_train, y_test = y[:,:-20], y[:,-20:]

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    # print(x_train, x_test)
    # print(y_train, y_test)

    LR = LinearRegression(x_train, y_train)
    LR.Hypothesis()
    print(LR.h.shape)                           # ->(422, 1)

    LR.CostFunction()
    print(LR.J)                                 # ->int

    LR.Derivative()
    print(LR.D0, LR.D1)

    # LR.getParam()
    tic = time.time()
    for i in range(50000):
        LR.UpdateParam()
        #print(LR.temp0, LR.temp1)
        #print(LR.D0, LR.D1)
        # LR.getParam()
    toc = time.time()
    print(str(1000*(toc-tic)) + 'ms')

    PLT = Visualize()

    LR.Hypothesis()
    PLT.plot(x_train, LR.h)

    PLT.scatter(x_train, y_train.T)
    # PLT.axis()
    PLT.cross()
    PLT.show()



########
main()####
########