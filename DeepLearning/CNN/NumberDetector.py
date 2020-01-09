#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Wyundi

import math
import imp
from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self):
        self.__datasets = datasets.load_digits()
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

    def getDataset(self):
        return self.__datasets

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

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.where(z<0, 0, z)

def Softmax(z):
    max = np.max(z,axis = 1,keepdims = True)
    h = (np.exp(z - max)) / np.sum(np.exp(z - max), axis = 1, keepdims = True)
    return h

def CorssEntropy(A, target):
    # print('CorssEntropy')
    A = np.where(A == 0, 1e-8, A)
    Loss = - target * np.log(A)
    
    return Loss

def Conv(img, ParamConv):
    # print('Conv')

    padding, kernal, b_conv, stride = ParamConv

    # print("img[0]:\n", img[0][0])
    '''
    if kernal.shape[0] != 1:
        print(img[0])

        print(kernal[0][0])
        print(kernal[0][1])
        print(kernal[0][2])
    '''
    # Parameter
    m, f_in, img_h, img_w = np.shape(img)
    f_in, f_out, k_h, k_w = np.shape(kernal)
    n_h = int((img_h - k_h + 2*padding) // stride + 1)
    n_w = int((img_w - k_w + 2*padding) // stride + 1)
    
    n = np.array([n_h, n_w])

    # Padding
    img_pad = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    # print("img_pad:\n", img_pad[0][0])

    # im2col
    img2col_h = n_h * n_w
    img2col_w = k_h * k_w
    img_im2col = np.zeros((m, f_in, img2col_h, img2col_w))

    for i in range(n_h):
        for j in range(n_w):
            img_im2col[..., i*n_h+j, :] = img_pad[..., i:i+k_h, j:j+k_w].reshape(m, f_in, k_h*k_w)

    # print("img_im2col:\n", img_im2col[0][0])
    # print(img_im2col.shape)

    kernal = kernal.reshape(f_in, f_out, k_h*k_w, 1)

    # print("kernal:\n", kernal[0][0])

    img_conv = np.zeros((m, f_out, n_h, n_w))

    for i in range(f_in):
        for j in range(f_out):
            img_conv[:, j] += np.dot(img_im2col[:, i], kernal[i, j]).reshape(m, n_h, n_w)

    for j in range(f_out):
        img_conv[:, j] = img_conv[:, j] + b_conv[j]
    
    '''
    if kernal.shape[0] != 1:
        print("b_conv:\n", b_conv)
        print("img_conv:\n", img_conv[0][0])
        print("img_conv:\n", img_conv[0][1])
        print("img_conv:\n", img_conv[0][2])
    '''

    return img_conv, img_im2col

def Conv_bw(img, img_bw, img_im2col, ParamConv_bw):
    # print('Conv_bw')

    # print("img_bw[0][0]:\n", img_bw[0][0])

    padding, kernal, b_conv, stride, alpha = ParamConv_bw
    kernal_shape = np.shape(kernal)
    b_conv_shape = np.shape(b_conv)
    img_bw_shape = np.shape(img_bw)
    f_in, f_out, k_h, k_w = kernal_shape
    m, f_out, ib_h, ib_w = img_bw_shape

    img_dX = np.zeros((m, 1, ib_h, ib_w))
    img_temp = np.zeros((m, 1, ib_h, ib_w))
    dK = np.zeros((f_in, 1, k_h, k_w))
    db = np.zeros(b_conv.shape)
    b = np.zeros(b_conv.shape)

    kernal_r = np.array(kernal)
    # print("kernal:\n", kernal[0][0])
    # print(kernal_r[0][5])
    # 翻转kernal
    temp = np.eye(kernal_r.shape[2])[::-1]
    for i in range(kernal_r.shape[0]):
        for j in range(kernal_r.shape[1]):
            kernal_r[i, j] = np.dot(temp, np.dot(kernal_r[i, j], temp))
    # kernal_r = kernal_r.transpose(1, 0, 2, 3)

    # print(kernal_r[5][0])
    # print("kernal_r:\n", kernal_r[0][0])

    for i in range(f_out):
        ParamConv = padding, kernal_r[0, i].reshape(1, 1, k_h, k_w), b, stride
        img_temp, img_im2colbw = Conv(img_bw[:, i].reshape(m, 1, ib_h, ib_w), ParamConv)
        img_dX += img_temp
    img_dX = np.tile(img_dX, (1, f_in, 1, 1))

    # print("img_dX[0]:\n", img_dX[0][0])

    img_bw_col = img_bw.reshape(m, f_out, ib_h * ib_w, 1)
    img_im2col_T = img_im2col.transpose(0, 1, 3, 2)
    '''
    for i in range(f_in):
        for j in range(f_out):
            for mm in range(m):
                dK[mm, i, j] = np.dot(img_im2col_T[mm, i], img_bw_col[mm, j]).reshape(k_h, k_w)
    '''

    img = np.pad(img, ((0,0), (0, 0), (padding, padding),(padding, padding)), 'constant')
    for mm in range(m):
        for x in range(f_in):
            for nn in range(f_out):
                for i in range(ib_h):
                    for j in range(ib_w):
                        dK[x, 0] += img[mm, x, i*stride:i*stride+k_h, j*stride:j*stride+k_w] * img_bw[mm, nn, i, j]
    dK = 1/m * np.tile(dK, (1, f_out, 1, 1))
    print(dK[0][0])

    # print(img_im2col_T[0][0])
    # print(img_bw[0][0])
    # dK = 1/m * np.sum(dK, axis = 0).reshape(kernal_shape)
    # print("dK:\n", dK[0][0])
    # print("kernal:\n", kernal[0][0])
    
    db = np.sum(img_bw, axis = (0, 2, 3)).reshape(b_conv_shape)

    kernal = kernal - alpha * dK
    b_conv = b_conv - alpha * db

    cache = (kernal, b_conv)
    # print(img_Convbw.shape)
    return img_dX, cache

def Pooling(img, stride):
    # print('Pooling')

    # print("img_shape:\n", img.shape)
    # print("img[0]:\n", img[0][0])

    in_shape = np.shape(img)
    m, f, img_h, img_w = in_shape
    s = stride

    H_out = math.ceil(img_h/s)
    W_out = math.ceil(img_w/s)
    out_shape = np.array([m, f, H_out, W_out])

    pool = np.zeros((m, f, s, s))
    img_pool = np.zeros(out_shape)

    for i in range(H_out):
        for j in range(W_out):
            pool = img[..., i*s:(i+1)*s, j*s:(j+1)*s]
            img_pool[..., i, j] = np.max(pool, axis = (2,3))

    # print("img_pool:\n", img_pool[0][0])

    return img_pool

def Pooling_bw(img, img_dX, stride):
    # print('Pooling_bw')

    # print("img[0][0]:\n", img[0][0])
    # print("img_dX[0][0]:\n", img_dX[0][0])

    in_shape = np.shape(img_dX)
    out_shape = np.shape(img)
    m, f, img_h, img_w = out_shape
    s = stride
    pool_h = math.ceil(img_h/s)
    pool_w = math.ceil(img_w/s)

    pool = np.zeros((m, f, s, s))
    img_pool_bw = np.zeros(out_shape)
    
    for i in range(pool_h):
        for j in range(pool_w):
            pool = img[..., i*s:(i+1)*s, j*s:(j+1)*s]
            mask = (pool == np.max(pool, axis = (2, 3))[..., np.newaxis, np.newaxis])
            img_pool_bw[..., i*s:(i+1)*s, j*s:(j+1)*s] = mask * img_dX[..., i, j][..., np.newaxis, np.newaxis]

    # print("img_pool_bw[0][0]:\n", img_pool_bw[0][0])

    return img_pool_bw

def FC(img, ParamFC):
    # print('FC')

    # print("img[0]:\n", img[0])

    W1, W2, W3, B1, B2, B3 = ParamFC

    Z1 = np.dot(img, W1) + B1.T
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + B2.T
    A2 = ReLU(Z2)
    Z3 = np.dot(A2, W3) + B3.T
    A3 = Softmax(Z3)

    '''
    # print(W1.shape, B1.shape)
    print("W1:\n", W1[:, 0])
    print("W2:\n", W2[:, 0])
    print("W3:\n", W3[:, 0])
    print("B1:\n", B1)
    print("B2:\n", B2)
    print("B3:\n", B3)
    print("Z1:\n", Z1[0])
    print("Z2:\n", Z2[0])
    '''
    # print("Z3:\n", Z3[0])
    # print("A3:\n", A3[0])
    # print("MAX(A3):", np.where(A3[0] == np.max(A3[0])))
    
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return cache

def FC_bw(img, target, ParamFC, ParamFC_rt, ParamFC_bw):
    # print('FC_bw')

    W1, W2, W3, B1, B2, B3 = ParamFC
    Z1, A1, Z2, A2, Z3, A3 = ParamFC_rt
    m, n, Layer_FC, dX, J, J_dv, Loss, alpha = ParamFC_bw

    Loss = CorssEntropy(A3, target)

    J0 = J
    J = 1/m * np.sum(Loss)

    # print("target:\n", target[0])
    print("Cost: ", J)
    
    J_dv = np.fabs(J-J0)

    dZ3 = (A3 - target)
    dW3 = 1/m * np.dot(A2.T, dZ3)
    dB3 = 1/m * np.sum(dZ3, axis = 0, keepdims = True).T

    dZ2 = np.dot(dZ3, W3.T) * np.where(Z2<0, 0, 1)
    dW2 = 1/m * np.dot(A1.T, dZ2)
    dB2 = 1/m * np.sum(dZ2, axis = 0, keepdims = True).T

    dZ1 = np.dot(dZ2, W2.T) * np.where(Z1<0, 0, 1)
    dW1 = 1/m * np.dot(img.T, dZ1)
    dB1 = 1/m * np.sum(dZ1, axis = 0, keepdims = True).T

    dX = np.dot(dZ1, W1.T) * np.where(img<0, 0, 1)

    W3 = W3 - alpha * dW3
    B3 = B3 - alpha * dB3
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1

    # print("dZ3:\n", dZ3[0])
    # print("dZ2:\n", dZ2[0])
    # print("dZ1:\n", dZ1[0])
    # print("dX:\n", dX[0])
    # print("img_dX:\n", img[0])

    cache = (dX, W1, B1, W2, B2, W3, B3, J, J_dv)
    return cache

def test(img, target, Param_conv1, Param_conv2, Param_FC):
    # Parameter
    Layer_conv = np.array([8, 16])
    PoolStride = np.array([2, 2])
    PoolingZoom1 = PoolStride[0]
    PoolingZoom2 = PoolStride[1]
    m = img.shape[0]
    n = int(Layer_conv[1] * img.shape[2]/(PoolingZoom1*PoolingZoom2) * \
                            img.shape[3]/(PoolingZoom1*PoolingZoom2))

    # test
    img_conv_1, img_im2col1 = Conv(img, Param_conv1)
    img_conv_1 = ReLU(img_conv_1)
    img_pool_1 = Pooling(img_conv_1, PoolStride[0])
    img_conv_2, img_im2col2 = Conv(img_pool_1, Param_conv2)
    img_conv_2 = ReLU(img_conv_2)
    img_pool_2 = Pooling(img_conv_2, PoolStride[1])

    img_forFC = img_pool_2.reshape(m, n)
    ParamFC_rt = FC(img_forFC, Param_FC)
    Z1, A1, Z2, A2, Z3, A3 = ParamFC_rt
    
    error = 0
    for i in range(img.shape[0]):
        if (np.where(target[i] == np.max(target[i]))) != (np.where(A3[i] == np.max(A3[i]))):
            error = error + 1

    print(100 - 100 * error/img.shape[0], '%')

# def loop(Param_conv1, Param_conv2, Param_FC):


def CNN(img, target):
    # Conv(8) - ReLU - Pooling - Conv(16) - ReLU - Pooling
    # FullyConnected: 16 - 16 - 10

    # Param of Conv and Pooling
    # print('CNN')

    m, c, img_h, img_w = np.shape(img)

    Layer_conv = np.array([8, 16])
    Padding = np.array([1, 1])
    Fliter = np.array([3, 3])
    ConvStride = np.array([1, 1])
    PoolStride = np.array([2, 2])

    ConvZoom1 = 1                               # ((n-f+2p)/s + 1)/n
    ConvZoom2 = 1                               # ((n-f+2p)/s + 1)/n
    PoolingZoom1 = PoolStride[0]
    PoolingZoom2 = PoolStride[1]

    Kernal_1 = np.random.rand(Layer_conv[0], Fliter[0], Fliter[0])
    Kernal_2 = np.random.rand(Layer_conv[1], Fliter[1], Fliter[1])

    Kernal_1 = np.tile(Kernal_1, (c, 1, 1, 1))
    Kernal_2 = np.tile(Kernal_2, (Layer_conv[0], 1, 1, 1))

    dK1 = np.random.rand(c, Layer_conv[0], Fliter[0], Fliter[0])
    dK2 = np.random.rand(Layer_conv[0], Layer_conv[1], Fliter[1], Fliter[1])

    b_conv1 = np.random.rand(Layer_conv[0], 1)
    b_conv2 = np.random.rand(Layer_conv[1], 1)

    dB_Conv1 = np.random.rand(Layer_conv[0], 1)
    dB_Conv2 = np.random.rand(Layer_conv[1], 1)

    # Param of FullyConnectedNeuraNet
    m = img.shape[0]
    n = int(Layer_conv[1] * img.shape[2]/(PoolingZoom1*PoolingZoom2) * \
                            img.shape[3]/(PoolingZoom1*PoolingZoom2))

    Layer_FC = np.array([16, 16, 10])

    W1 = np.random.rand(n, Layer_FC[0])
    W2 = np.random.rand(Layer_FC[0], Layer_FC[1])
    W3 = np.random.rand(Layer_FC[1], Layer_FC[2])

    dW1 = np.random.rand(n, Layer_FC[0])
    dW2 = np.random.rand(Layer_FC[0], Layer_FC[1])
    dW3 = np.random.rand(Layer_FC[1], Layer_FC[2])

    B1 = np.random.rand(Layer_FC[0], 1)
    B2 = np.random.rand(Layer_FC[1], 1)
    B3 = np.random.rand(Layer_FC[2], 1)

    dB1 = np.random.rand(Layer_FC[0], 1)
    dB2 = np.random.rand(Layer_FC[1], 1)
    dB3 = np.random.rand(Layer_FC[2], 1)

    img_dX = np.zeros([m, n])

    dX = np.zeros([m, n])

    Z1 = np.zeros([m, Layer_FC[0]])
    A1 = np.zeros([m, Layer_FC[0]])
    Z2 = np.zeros([m, Layer_FC[1]])
    A2 = np.zeros([m, Layer_FC[1]])
    Z3 = np.zeros([m, Layer_FC[2]])
    A3 = np.zeros([m, Layer_FC[2]])

    dZ1 = np.zeros([m, Layer_FC[0]])
    dA1 = np.zeros([m, Layer_FC[0]])
    dZ2 = np.zeros([m, Layer_FC[1]])
    dA2 = np.zeros([m, Layer_FC[1]])
    dZ3 = np.zeros([m, Layer_FC[2]])
    dA3 = np.zeros([m, Layer_FC[2]])

    Loss = np.zeros([m, 1])
    J = 0
    J_dv = 0

    alpha = 0.3
    i = 0

    # All Parameter
    Param_conv1 = (Padding[0], Kernal_1, b_conv1, ConvStride[0])
    Param_conv2 = (Padding[1], Kernal_2, b_conv2, ConvStride[1])

    Param_FC = (W1, W2, W3, B1, B2, B3)
    ParamFC_rt = (Z1, A1, Z2, A2, Z3, A3)
    ParamFC_bw = (m, n, Layer_FC, dX, J, J_dv, Loss, alpha)
    ParamUpdate_FCbw = (img_dX, W1, B1, W2, B2, W3, B3, J, J_dv)

    ParamConv2_bw = (Padding[1], Kernal_2, b_conv2, ConvStride[1], alpha)
    ParamUpdate_Conv2bw = (Kernal_2, b_conv2)
    ParamConv1_bw = (Padding[0], Kernal_1, b_conv1, ConvStride[0], alpha)
    ParamUpdate_Conv1bw = (Kernal_1, b_conv1)

    # test(img, target, Param_conv1, Param_conv2, Param_FC)

    while(True):
        '''
        if i == 1:
            break
        '''

        i = i + 1
        # All Parameter
        Param_conv1 = (Padding[0], Kernal_1, b_conv1, ConvStride[0])
        Param_conv2 = (Padding[1], Kernal_2, b_conv2, ConvStride[1])

        Param_FC = (W1, W2, W3, B1, B2, B3)
        ParamFC_rt = (Z1, A1, Z2, A2, Z3, A3)
        ParamFC_bw = (m, n, Layer_FC, dX, J, J_dv, Loss, alpha)
        ParamUpdate_FCbw = (img_dX, W1, B1, W2, B2, W3, B3, J, J_dv)

        ParamConv2_bw = (Padding[1], Kernal_2, b_conv2, ConvStride[1], alpha)
        ParamUpdate_Conv2bw = (Kernal_2, b_conv2)
        ParamConv1_bw = (Padding[0], Kernal_1, b_conv1, ConvStride[0], alpha)
        ParamUpdate_Conv1bw = (Kernal_1, b_conv1)

        # test-Forword
        img_conv_1, img_im2col1 = Conv(img, Param_conv1)
        img_conv_1_ReLU = ReLU(img_conv_1)
        img_pool_1 = Pooling(img_conv_1_ReLU, PoolStride[0])
        img_conv_2, img_im2col2 = Conv(img_pool_1, Param_conv2)
        img_conv_2_ReLU = ReLU(img_conv_2)
        img_pool_2 = Pooling(img_conv_2_ReLU, PoolStride[1])

        img_forFC = img_pool_2.reshape(m, n)
        ParamFC_rt = FC(img_forFC, Param_FC)

        # test-Backword
        ParamUpdate_FCbw = FC_bw(img_forFC, target, Param_FC, ParamFC_rt, ParamFC_bw)
        img_dX, W1, B1, W2, B2, W3, B3, J, J_dv = ParamUpdate_FCbw
        # print("dX:\n", dX)
        img_dX = img_dX.reshape(np.shape(img_pool_2))
        # print("img_dX_reshape:\n", img_dX)

        img_forConvbw2 = Pooling_bw(img_conv_2, img_dX, PoolStride[1])
        img_forConvbw2 = img_forConvbw2 * np.where(img_conv_2<0, 0, 1)
        img_forPoolbw1, ParamUpdate_Conv2bw = Conv_bw(img_pool_1, img_forConvbw2, img_im2col2, ParamConv2_bw)
        Kernal_2, b_conv2 = ParamUpdate_Conv2bw

        img_forConvbw1 = Pooling_bw(img_conv_1, img_forPoolbw1, PoolStride[0])
        img_forConvbw1 = img_forConvbw1 * np.where(img_conv_1<0, 0, 1)
        img_ending, ParamUpdate_Conv1bw = Conv_bw(img, img_forConvbw1, img_im2col1, ParamConv1_bw)
        Kernal_1, b_conv1 = ParamUpdate_Conv1bw
        
        if J_dv <= 0.000000000000001:
            break
        
    Param_conv1 = (Padding[0], Kernal_1, b_conv1, ConvStride[0])
    Param_conv2 = (Padding[1], Kernal_2, b_conv2, ConvStride[1])

    Param_FC = (W1, W2, W3, B1, B2, B3)
    
    test(img, target, Param_conv1, Param_conv2, Param_FC)


    return Param_conv1, Param_conv2, Param_FC

def main():
    Data = GetData()
    dataset = Data.getDataset()
    data = Data.getData()
    img = dataset.images
    img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
    target = dataset.target
    target = target.reshape(target.shape[0], 1)
    
    img = img / np.max(img)
    target_trans = np.zeros((img.shape[0], 10))
    for i in range(img.shape[0]):
        target_trans[i, target[i]] = 1
    
    # print(dataset.keys())

    x_train, x_test, y_train, y_test = train_test_split(img, target_trans, test_size = 0.2)
    
    Param_conv1, Param_conv2, Param_FC = CNN(x_train, y_train)

    test(x_test, y_test, Param_conv1, Param_conv2, Param_FC)

    # Param_conv1, Param_conv2, Param_FC = CNN(x_train[0].reshape(1,1,8,8), y_train[0])
    # plt.imshow(dataset.images[0])
    # plt.show()

main()