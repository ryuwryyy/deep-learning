import sys, os
sys.path.append(os.pardir)
import numpy as np
from common layers import *
from common.gradient import numerical_gradient
from collections import OrderDict

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

    self.params = {}
    self.params['W1'] = weight_init_std * np.random/randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b2'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    return x

  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)

  def accuracy(self, x, y)
    y = self.predict(x)
    y = np.argmax(t, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
  return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lamdba W:self.loss(x, t)

    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W1'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b2'])
    return grads

  def gradient(self, x, t):
    self.loss(x, t)

    dout = 1
    dout= self.lastLater.backward(dout)

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

    import sys, os
    sys.path.append(os, pardir)
    import numpy as np
    from dataset.mnist import load_mnist
    from two_layer_net import TwoLayerNet

    (x_train, t_train), (x_test, T_test) = load_nist(nomalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
      batch_mask = np.random.choice(train_size, batch_size)
      x_batch = x_train[batch_mask]
      t_batch = t_train[batch_mask]

      grad = network.gradient(x_batch, t_batch)

      for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

      loss = network.loss(x_batch, t_batch)
      train_loss_list.append(loss)

      if i % iter_per_epoch ==0:
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ('W1','b1', 'W2', 'b2'):
          network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

#1 ニューラルネットワーク
import
from collections import OrderedDict

class TwoLayerNet
  def __init__():

  def predict(self, x):
  return x

  def loss(self, x, t):
  return self.lastLayer.forward(y, t)

  def accuracy(self, x, t):
  return accuracy

  def numerical_gradient(self, x, t):
  return grads

  def gradient(self, x, t):
  return grads

#2 勾配確認
import sys, os
sys.path.append(os, pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

##データの読み込み
(x_train, t_train), (x_test, t_train) = load_mnist(nomalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
  diff = np.average
  print(key + ":" + str(diff))

#3 学習
import sys, os
sys.path.append(os, pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(nomalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size=10)

iters_num =10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate 0.1

train_loss_list = []
train_acc_list=[]
test_acc_list = []

#訓練データをすべて使い切ったときの回数
iter_per_epock = max(train_size / batch_size. 1)

for i in range(iters_num):
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

#誤差逆伝搬法によって勾配をもとめる
grad = network.gradient(x_batch, t_batch)

#更新
for key in ('W1','b1','W2','b2'):
  network.params[key] -= learning_rate * grad[key]
loss = network.loss(x_batch, t_batch)
train_loss_list.append(loss)
if i % iter_per_epoch == 0:
    print(train_acc, test_acc)