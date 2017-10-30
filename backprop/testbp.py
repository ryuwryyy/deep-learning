# import numpy as np
# import random

# def XOR(x, y):
#     random_result = random.choice([0,1])
#     if random_result == 1:
#         return [1, 0]
#     else:
#         return [0, 1]

# def cross_entropy_error(y, t):
#         delta = 1e-7
#         return -np.sum(t * np.log(y + delta))

# def nueral_network(x, y):
#     X = [x1, x2]
#     W =[w1, w2]
#     B = [b1, b2]
#     A = [a1, a2]
#     Z = [z1, z2]
#     Y = [y1, y2]
#     network = [[b1, W1],[b2, W2]]
#     np.random.randn(network)
#     a = np.dot(network, X) + B
# #     Z=sigmoid(A)
#     return a
# #ニューラルネットワーク
# def numerical_gradient(x, y):
    
#     h=1e-4
#     grad = np.zeros_like(x)
#     for idx in range(len(size)):
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h
#         fxh1
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
        
#         grad[idx] = (fxh1 - fxh2) / (2*h)
#         x[idx] = tmp_val
        
#     return x
    
# def gradient_descent(nueral_network, init_x, lr=0.01, step_num=1000):
#     x = init_x
    
#     for i in range(step_num):
#         grad = numerical_gradient(nueral_network, x)
#         x -= lr * grad
#     return x

# gradient_descent(nueral_network, init_x, lr=0.01, step_num=1000

# import sys, os
sys.path.append(os, pardir)
import numpy as np
from common layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params={}
        self.params['W1'] = weighr_init_std * np.random.randn(input_size, hidden_size)
        self/params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weighr_init_std * np.random.randn(hidden_size, output_size)
        self.params = np.zeros(output_size)

        self.layers = OrderedDict()
        self,layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu']=Relu()
        self,layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #lastkayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self,layers.vakue():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(t, axis=1)
        if t.ndim != 1:t = np.argmax(t, axis=1):

            accuracy = np.sum(y==t) / float(x.shap[0])
    return accuracy

    def numerical_gradient(self, x,t):
        loss_W = kambda W:self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
          grads['b2'] = numerical_gradient(losss_W, self.params['b2'])
    return grads

    def gradient(self, x, t):
        self.loss(x,t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.value())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
      grads = {}
      grads['W1'] = self.layers['Affine1'].dw
      grads['b1'] = self.layers['Affine1'].db
      grads['W2'] = self.layers['Affine2'].dw
      grads['b2'] = self.layers['Affine2'].db

      return grads

#勾配の確認
network = TwoLayerNet(input_size = , hidden_size= , output_size, )

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backpop[key] - grad_numerical[key]))
    print(key + str(diff))




#誤差逆伝搬法の学習
network 0 TwoLayerNet(input_size, hidden_size, output_size)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

train_loss_list =[]
train_acc_list=[]
test_acc=[]

iter_per_eporch = max(train_size/batich_size, 1)

for i in range(iters_num):
    batch_masl = np.random.shoice(train_size, batch_size)
    x_batch = x_train [batch_nask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -=learning_rate* grad[key]
    loss = network_loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_batxh, t_batch)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(trai_acc, test_acc)