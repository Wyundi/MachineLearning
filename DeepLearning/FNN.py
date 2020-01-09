import os
import struct
import numpy as np

# mnist datasets
path = '/home/wyundi/Project/Git/MachineLearning/DeepLearning/mnist/'

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

images, labels_num = load_mnist(path)
images_test, labels_test = load_mnist(path, 't10k')

labels = np.zeros((labels_num.shape[0], np.max(labels_num) + 1))
row = np.arange(0, labels_num.shape[0])
labels[row, labels_num] = 1

x = images[0:500, :]
y = labels[0:500, :]

print(x.shape)

print(images.shape, labels.shape)
print(images_test.shape, labels_test.shape)

# NeuralNet
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.where(z<0, 0, z)

def LeakyReLU(z):
    return np.where(z<0, 0.01 * z, z)

def Softmax(z):
    max = np.max(z,axis = 1,keepdims = True)
    h = (np.exp(z - max)) / np.sum(np.exp(z - max), axis = 1, keepdims = True)
    return h

m = x.shape[0]
n = np.array([x.shape[1], 16, 16, y.shape[1]])

W1 = np.random.rand(n[0], n[1])
W2 = np.random.rand(n[1], n[2])
W3 = np.random.rand(n[2], n[3])

B1 = np.random.rand(n[1], 1)
B2 = np.random.rand(n[2], 1)
B3 = np.random.rand(n[3], 1)

J0 = 0
J = 0
J_dv = abs(J0 - J)

alpha = 0.03

time = 10

while(True):
    Z1 = np.dot(x, W1) + B1.T
    A1 = LeakyReLU(Z1)
    Z2 = np.dot(A1, W2) + B2.T
    A2 = LeakyReLU(Z2)
    Z3 = np.dot(A2, W3) + B3.T
    A3 = Softmax(Z3)

    A3_Loss = np.where(A3 == 0, 1e-8, A3)

    Loss = - y * np.log(A3_Loss)

    J0 = J
    J = 1/m * np.sum(Loss)
    J_dv = np.fabs(J0 - J)
    print(J)
    
    dZ3 = A3 - y
    dW3 = 1/m * np.dot(A2.T, dZ3)
    dB3 = 1/m * np.sum(dZ3, axis = 0, keepdims = True).T

    dZ2 = np.dot(dZ3, W3.T) * np.where(Z2<0, 0.01, 1)
    dW2 = 1/m * np.dot(A1.T, dZ2)
    dB2 = 1/m * np.sum(dZ2, axis = 0, keepdims = True).T

    dZ1 = np.dot(dZ2, W2.T) * np.where(Z1<0, 0.01, 1)
    dW1 = 1/m * np.dot(x.T, dZ1)
    dB1 = 1/m * np.sum(dZ1, axis = 0, keepdims = True).T
    '''
    print(np.where(A3[0] == np.max(A3[0])), labels_num[0])
    
    print(A3[0], labels[0])
    print(dZ3[0])
    '''

    W3 = W3 - alpha * dW3
    B3 = B3 - alpha * dB3
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1

    if J_dv <= 0.0000001:
        break
    '''
    time -= 1
    if time == 0:
        break
    '''