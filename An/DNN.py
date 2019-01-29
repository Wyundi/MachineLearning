#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import imp
import numpy as np
from sklearn.model_selection import train_test_split

Feature = 7

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.where(z<0, 0, z)

def Softmax(z):                                         ######## that's it
    max = np.max(z,axis = 1,keepdims = True)
    h = (np.exp(z - max)) / np.sum(np.exp(z - max), axis = 1, keepdims = True)
    return h

def CorssEntropy(A, target):
    # print('CorssEntropy')
    temp = np.where(A * target == 0, A + 0.05, A * target)
    Loss = -np.log(temp)
    
    return Loss

def NeuralNet(X, y):
    m = X.shape[0]
    n = X.shape[1]
    Layer = np.array([4, 6, 6, 4, Feature])

    W1 = np.random.rand(n, Layer[0])
    W2 = np.random.rand(Layer[0], Layer[1])
    W3 = np.random.rand(Layer[1], Layer[2])
    W4 = np.random.rand(Layer[2], Layer[3])
    W5 = np.random.rand(Layer[3], Layer[4])

    B1 = np.random.rand(Layer[0], 1)
    B2 = np.random.rand(Layer[1], 1)
    B3 = np.random.rand(Layer[2], 1)
    B4 = np.random.rand(Layer[3], 1)
    B5 = np.random.rand(Layer[4], 1)

    Z1 = np.zeros([m, Layer[0]])
    A1 = np.zeros([m, Layer[0]])
    Z2 = np.zeros([m, Layer[1]])
    A2 = np.zeros([m, Layer[1]])
    Z3 = np.zeros([m, Layer[2]])
    A3 = np.zeros([m, Layer[2]])
    Z4 = np.zeros([m, Layer[3]])
    A4 = np.zeros([m, Layer[3]])
    Z5 = np.zeros([m, Layer[4]])
    A5 = np.zeros([m, Layer[4]])

    Loss = np.zeros([m, 1])
    J = 0

    alpha = 0.03
    i = 0
    
    while(True):
        i = i + 1
        # 正向传播
        Z1 = np.dot(X, W1) + B1.T
        A1 = ReLU(Z1)
        Z2 = np.dot(A1, W2) + B2.T
        A2 = ReLU(Z2)
        Z3 = np.dot(A2, W3) + B3.T
        A3 = ReLU(Z3)
        Z4 = np.dot(A3, W4) + B4.T
        A4 = ReLU(Z4)
        Z5 = np.dot(A4, W5) + B5.T
        A5 = Softmax(Z5)

        # 反向传播
        Loss = CorssEntropy(A5, y)

        J0 = J
        J = 1/m * np.sum(Loss)
        print(J)
        J_dv = np.fabs(J-J0)

        # dx 和 x 的维度相同
        dZ5 = (A5 - y)
        dW5 = 1/m * np.dot(A4.T, dZ5)
        dB5 = 1/m * np.sum(dZ5, axis = 0, keepdims = True).T

        dZ4 = np.dot(dZ5, W5.T) * np.where(Z4<0, 0, 1)
        dW4 = 1/m * np.dot(A3.T, dZ4)
        dB4 = 1/m * np.sum(dZ4, axis = 0, keepdims = True).T

        dZ3 = np.dot(dZ4, W4.T) * np.where(Z3<0, 0, 1)
        dW3 = 1/m * np.dot(A2.T, dZ3)
        dB3 = 1/m * np.sum(dZ3, axis = 0, keepdims = True).T

        dZ2 = np.dot(dZ3, W3.T) * np.where(Z2<0, 0, 1)
        dW2 = 1/m * np.dot(A1.T, dZ2)
        dB2 = 1/m * np.sum(dZ2, axis = 0, keepdims = True).T

        dZ1 = np.dot(dZ2, W2.T) * np.where(Z1<0, 0, 1)
        dW1 = 1/m * np.dot(X.T, dZ1)
        dB1 = 1/m * np.sum(dZ1, axis = 0, keepdims = True).T

        #参数更新
        W5 = W5 - alpha * dW5
        B5 = B5 - alpha * dB5
        W4 = W4 - alpha * dW4
        B4 = B4 - alpha * dB4
        W3 = W3 - alpha * dW3
        B3 = B3 - alpha * dB3
        W2 = W2 - alpha * dW2
        B2 = B2 - alpha * dB2
        W1 = W1 - alpha * dW1
        B1 = B1 - alpha * dB1

        if J_dv <= 0.0000001:
            break
    
    print("i = ", i)
    return W5, B5, W4, B4, W3, B3, W2, B2, W1, B1

def loop(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    W5, B5, W4, B4, W3, B3, W2, B2, W1, B1 = NeuralNet(x_train, y_train)
    
    Z1 = np.dot(x_test, W1) + B1.T
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + B2.T
    A2 = ReLU(Z2)
    Z3 = np.dot(A2, W3) + B3.T
    A3 = ReLU(Z3)
    Z4 = np.dot(A3, W4) + B4.T
    A4 = ReLU(Z4)
    Z5 = np.dot(A4, W5) + B5.T
    A5 = Softmax(Z5)

    m = 0
    for i in range(x_test.shape[0]):
        print(np.where(y_test[i] == np.max(y_test[i])))
        print(np.where(A5[i] == np.max(A5[i])))
        if (np.where(y_test[i] == np.max(y_test[i]))) != (np.where(A5[i] == np.max(A5[i]))):
            m = m + 1
        print(A5[i], '\n', y_test[i])
    print("m = ", m)
    return m

def main():
    file = open('/home/wyundi/Project/Code/Python/DeepLearning/An/data', 'r')
    data = file.readlines()
    file.close()
    for n in range(len(data)):
        data[n] = data[n].replace(' \n', '')
        data[n] = data[n].split(' \t')
    data = np.array(data)
    # print(data)

    x = data[:, :-1].astype('float64')
    y = data[:, -1].reshape(data.shape[0], 1).astype('int64')
    # print(x.shape, y.shape)

    y_trans = np.zeros(x.shape)

    for i in range(y.shape[0]):
        y_trans[i, y[i]] = 1

    time = 1
    while(loop(x, y_trans) >= 2):
        time = time + 1
        print('motherfucker')
    print('time = ', time)

######
main()
######