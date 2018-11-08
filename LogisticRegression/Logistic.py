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

# 分割数据的模块，把数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self):
        # diabetes datasets
        self.__datasets = datasets.load_iris()
        self.__data = self.__datasets.data
        self.__target = np.array([self.__datasets.target])
        self.TrainTest()

    def oneFeaturn(self, n):
        self.__data = self.__data[:, np.newaxis, n]
        self.TrainTest()
        return self.__data

    def multiFeaturn(self, n, m):
        self.__data = self.__data[:, n:m]
        self.TrainTest()
        return self.__data

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
        self.__data_train, self.__data_test, self.__target_train, self.__target_test = \
        train_test_split(self.__data, self.__target.T, test_size=0.1)

    def getData(self):
        return self.__data

    def getTarget(self):
        return self.__target

    def getTrainData(self):
        return self.__data_train

    def getTestData(self):
        return self.__data_test

    def getTrainTarget(self):
        return self.__target_train.T

    def getTestTarget(self):
        return self.__target_test.T

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

class Logistic():
    def __init__(self, X, y):
        self.m = X.shape[0]                     # number of examples
        self.n = X.shape[1]                     # number of features
        self.x_extra = np.ones([self.m, 1])
        self.data_X = np.column_stack((self.x_extra, X))
        self.data_y = y.T
        
        self.m = self.data_X.shape[0]
        self.n = self.data_X.shape[1]

        self.__theta = np.random.rand(self.n, 1)
        self.__alpha = 0.01

        self.h = np.zeros([self.m, 1])

        self.J = 0
        self.J0 = 0
        self.J_dv = 0

        self.D = np.zeros([self.n, 1])
        self.temp = np.zeros([self.n, 1])

    def modifyParam(self, alpha):
        self.__alpha = alpha

    def getTheta(self):
        return self.__theta

    def Hypothesis(self):
        self.h = self.sigmoid(np.dot(self.data_X, self.__theta))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def CostFunction(self):
        self.Hypothesis()
        self.J0 = self.J
        self.J = 1/(2*self.m) * np.sum(np.power((self.h - self.data_y), 2))
        self.J_dv = self.J0 - self.J     

    def drawCostFunction(self):

        print("motherfucker")
        
    def Derivative(self):
        self.CostFunction()
        for i in range(self.n):
            self.D[i] = 1/self.m * np.sum((self.h - self.data_y) * self.data_X[:, i].reshape(self.m, 1))

    def UpdateParam(self):
        self.Derivative()
        self.__theta = self.__theta - self.__alpha * self.D

def main():
    Data = GetData()
    Data.multiFeaturn(0, 2)
    Data.FeatureScaling()
    # Data.MeanNormalization()

    x = Data.getData()
    y = Data.getTarget()

    # 将数据分为训练集和测试集
    x_train, x_test = Data.getTrainData(), Data.getTestData()
    y_train, y_test = Data.getTrainTarget(), Data.getTestTarget()

    # print("x_train/test:", x_train.shape, x_test.shape)
    # print("y_train/test:", y_train.shape, y_test.shape)
    
    y0 = np.zeros([y_train.shape[0], y_train.shape[1]])
    y1 = np.ones([y_train.shape[0], y_train.shape[1]])
    y2 = 2 * y1
    y0_train = (y_train == y0).astype('int')
    y1_train = (y_train == y1).astype('int')
    y2_train = (y_train == y2).astype('int')

    y0 = np.zeros([y_test.shape[0], y_test.shape[1]])
    y1 = np.ones([y_test.shape[0], y_test.shape[1]])
    y2 = 2 * y1
    y0_test = (y_test == y0).astype('int')
    y1_test = (y_test == y1).astype('int')
    y2_test = (y_test == y2).astype('int')

    LR = Logistic(x_train, y0_train)
    i = 0
    while(True):
        i += 1
        LR.UpdateParam()
        # theta0 = LR.getTheta().T
        # print(LR.J, LR.J_dv)
        # print(theta, LR.D.T)
        if abs(LR.J_dv) <= 0.0000001:
            break
    
    print(i)
    theta0 = LR.getTheta()

    LR = Logistic(x_train, y1_train)
    i = 0
    while(True):
        i += 1
        LR.UpdateParam()
        # theta0 = LR.getTheta().T
        # print(LR.J, LR.J_dv)
        # print(theta, LR.D.T)
        if abs(LR.J_dv) <= 0.0000001:
            break
    
    print(i)
    theta1 = LR.getTheta()

    LR = Logistic(x_train, y2_train)
    i = 0
    while(True):
        i += 1
        LR.UpdateParam()
        # theta0 = LR.getTheta().T
        # print(LR.J, LR.J_dv)
        # print(theta, LR.D.T)
        if abs(LR.J_dv) <= 0.0000001:
            break
    
    print(i)
    theta2 = LR.getTheta()

    PLT = Visualize()

    for i in range(y_train.shape[1]):
        if y_train[0, i] == 0:
            PLT.scatter(x_train[i, 0], x_train[i ,1], 'green')
        elif y_train[0, i] == 1:
            PLT.scatter(x_train[i, 0], x_train[i ,1], 'yellow')
        else:
            PLT.scatter(x_train[i, 0], x_train[i ,1], 'red')

    x1_t = x_train
    x2_t0 = - (theta0[0] + theta0[1] * x1_t) / theta0[2]
    x2_t1 = - (theta1[0] + theta1[1] * x1_t) / theta1[2]
    x2_t2 = - (theta2[0] + theta2[1] * x1_t) / theta2[2]
    PLT.plot(x1_t, x2_t0)
    PLT.plot(x1_t, x2_t1)
    PLT.plot(x1_t, x2_t2)
    PLT.show()


main()