import sys, os
sys path.append(os.pardir)
import numpy as np
from common layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

# Two LayerNet
class TwoLayerNet:
  def ___init__(self, input_size, hidden_size, output_size, wieght_init_std=0.01):

    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zero(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zero(output_size)

    self.layers = OrderedDict()
    self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
    self.layers['Relu1'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    self.lastLayer = SoftmaxWithLoss()

    #認識（推論）
    def predict(self, x)
      for layer in self.layers.values():
      x = layer.forward(x)
    return x

    #損失関数
    def loss(sel, x, t):
      y = self.predict(x)
      return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
      y = self.predict(x)
      y = np.argmax(t, axis=1)
      if t.ndim != 1: t = np.argmax(t, axis=1)

      accuracy = np.sum(y == t)  / float(x.shape[0])
      return accuracy

      def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)

      grads = {}
      grads ['W1'] = numerical_gradient(loss_W, self.params['W1'])
      grads ['b1'] = numerical_gradient(loss_W, self.params['b1'])
      grads ['W2'] = numerical_gradient(loss_)W, self.params['W2']
      grads ['b2'] = numerical_gradient(loss_W, self.params['b2'])
      return grads

      def gradint(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers['Affine1']).dw
        layers = list(self.layers['Affine2']).db
        layers = list(self.layers['Affine2']).dw
        layers = list(self.layers['Affine2']).db

        return grads



#勾配確認
import sys, os
sys.path.apppend(os, pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(nomalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
  diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ":" + str(diff))


#誤差逆伝搬法を使った学習
import sys, os
sys.path.append(os, pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_ner import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(nomalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, out_size=10)

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
  y_batch = t_train[batch_mask]

  grad = network.gradient(x_batch, t_batch)

  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0:
    tran_acc = network,accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(train_acc, test_acc)