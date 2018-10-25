#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

# Multivariate Linear Regression

import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time
from mpl_toolkits.mplot3d import Axes3D

class GetData():
    def __init__(self):
        # diabetes datasets
        self.__diabetes = datasets.load_diabetes()
        self.__data = self.__diabetes.data
        self.__target = np.array([self.__diabetes.target])
        self.TrainTest()

    def oneFeaturn(self, n):
        self.__data = self.__data[:, np.newaxis, n]
        self.TrainTest()

    def FeatureScaling(self):
        max = np.max(self.__data, axis = 0)
        min = np.min(self.__data, axis = 0)
        for i in range(self.__data.shape[1]):
            for j in range(self.__data.shape[0]):
                self.__data[j, i] = self.__data[j, i]/(max[i] - min[i])

    def MeanNormalization(self):
        max = np.max(self.__data, axis = 0)
        min = np.min(self.__data, axis = 0)
        mean = np.mean(self.__data, axis = 0)
        for i in range(self.__data.shape[1]):
            for j in range(self.__data.shape[0]):
                self.__data[j, i] = (self.__data[j, i] - mean[i])/(max[i] - min[i])

    def TrainTest(self):
        self.__data_train, self.__data_test = self.__data[:-20], self.__data[-20:]
        self.__target_train, self.__target_test = self.__target[:,:-20], self.__target[:,-20:]

    def getData(self):
        return self.__data

    def getTarget(self):
        return self.__target

    def getTrainData(self):
        return self.__data_train

    def getTestData(self):
        return self.__data_test

    def getTrainTarget(self):
        return self.__target_train

    def getTestTarget(self):
        return self.__target_test

class LinearRegression():
    def __init__(self, X, y):
        self.m = X.shape[0]                     # number of examples
        self.n = X.shape[1]                     # number of features
        self.x_extra = np.ones([self.m, 1])
        self.data_X = np.column_stack((self.x_extra, X))
        self.data_y = y.T
        print(self.data_y.shape)
        self.m = self.data_X.shape[0]
        self.n = self.data_X.shape[1]

        self.__theta = np.random.rand(self.n, 1)
        self.__alpha = 0.01

        self.h = np.zeros([self.m, 1])

        self.J = 0
        self.J0 = 0
        self.J_dv = 0
        self.J_img = np.zeros([12000, 1])

        self.D = np.zeros([self.n, 1])
        self.temp = np.zeros([self.n, 1])
        print("alpha = ", self.__alpha)

    def modifyParam(self, alpha):
        self.__alpha = alpha

    def getTheta(self):
        return self.__theta

    def Hypothesis(self):
        self.h = np.dot(self.data_X, self.__theta)

    def CostFunction(self):
        self.Hypothesis()
        self.J0 = self.J
        self.J = 1/(2*self.m) * np.sum(np.power((self.h - self.data_y), 2))
        self.J_dv = self.J0 - self.J     

    def drawCostFunction(self):
        print(self.__theta.shape)
        self.theta1 = np.arange(-3000, 3000, 0.5)
        self.theta2 = np.arange(-3000, 3000, 0.5)
        self.theta1 = self.theta1.reshape([12000, 1])
        self.theta2 = self.theta2.reshape([12000, 1])
        self.theta0 = np.column_stack((self.theta1, self.theta2))
        print(self.theta0.shape)
        
        for i in range(12000):
            for j in range(12000):
                self.__theta = self.theta0[i]
                self.CostFunction()
                print(self.J)
                self.J_img[i] = self.J
        
        PLT = Visualize()
        PLT.plot(self.theta2, self.J_img)
        PLT.cross()
        PLT.show()

    def Derivative(self):
        self.CostFunction()
        for i in range(self.n):
            self.D[i] = 1/(2*self.m) * np.sum((self.h - self.data_y) * self.data_X[:][i])

    def UpdateParam(self):
        self.Derivative()
        self.temp = self.__theta - self.__alpha * self.D
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
    Data = GetData()
    Data.oneFeaturn(2)
    Data.FeatureScaling()
    # Data.MeanNormalization()
    x = Data.getData()
    y = Data.getTarget()

    # 将数据分为训练集和测试集
    x_train, x_test = Data.getTrainData(), Data.getTestData()
    y_train, y_test = Data.getTrainTarget(), Data.getTestTarget()

    # print("x_train/test:", x_train.shape, x_test.shape)
    # print("y_train/test:", y_train.shape, y_test.shape)

    ### 测试数据
    LR = LinearRegression(x_train, y_train)
    PLT = Visualize()

    i = 0

    LR.modifyParam(0.1)
    
    LR.UpdateParam()
    while(True):
        i += 1
        LR.UpdateParam()
        theta = LR.getTheta().T
        print(LR.J)
        print(theta)
        print(LR.D.T)
        if LR.D.any() <= 0.001:
            break

    '''
    LR.CostFunction()
    i = 0
    while(True):
        i += 1
        LR.UpdateParam()
        LR.CostFunction()
        print(LR.J_dv)
        if LR.J_dv <= 0.0000001:
            break
    
    print(i)

    PLT.plot(x_train, LR.h)
    PLT.scatter(x_train, y_train.T)

    # PLT.plot(x_train, LR.h)
    # PLT.scatter(x_test, y_test)

    # PLT.axis()
    PLT.cross()
    PLT.show()
    '''

######

    '''
    LR.Hypothesis()
    # print("h.shape:", LR.h.shape)                           # ->(422, 1)
    # print("h:", LR.h)

    LR.CostFunction()
    print("CostFunction J:", LR.J)

    LR.Derivative()
    
    ### 回归
    tic = time.time()

    for i in range(LR.iteration):
        LR.UpdateParam()
        LR.CostFunction()

    toc = time.time()
    print(str(1000*(toc-tic)) + 'ms')

    print("J.min: ", np.min(LR.J_iteration))
    
    # PLT.plot(x_train, LR.h)
    # PLT.scatter(x_train, y_train.T)

    # PLT.axis()
    PLT.cross()
    PLT.show()
    '''
######
main()