#バッチ処理
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
  x_batch = x[i:i+batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis=1)
  accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#損失関数
##lost function
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

##cross entropy function
def cross-entropy-error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))

#ミニバッジ学習
import sys, os
sys.path.append(os, pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
  load_mnist(nomalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_bath = t_train[batch_mask]

def numerical_diff(f, x):
  h = le-4
  return (f(x+h)) - f(x-h) / (2*h)

#勾配 gradient
def numerical_gradient(f, x):
  h = le-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp-val = x[idx]
    x[idex] = tmp_val + h
    fxh1 =f(x)

    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idex] = tmp_val
  return grad
#勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -=lr * grad
  return x

#勾配を求める実装
import sys, os
sys.path.append(os, pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
  def __init__(self):
    self.W = np.random.random.randn(2,3)
  def predict(self, x):
    return np.rdot(x, self.W)
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    return loss

#2層ニューラルネットワークのクラス
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

    self.prams = {}
    self.prams['W1'] = weight_init_std *\nnnp.random.randn(input_size, hiden_size)
    self.prams['b1'] = np.zeros(hidden_size)
    self.prams['W2'] = weight_init_std * \ np.random.randn(hidden_size, output_size)
    self.prams['b2'] = np.zeros(output_size)

  def predict(self, x):
    W1, W2 = self.prams['W1'], self.prams['W2']
    b1, b2 = self.prams['b1'], self.prams['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    y = softmax(a2)

    return y

  def loss(self, x, t):
    y = self.predient(x)

    return cross_entropy_error(y, t)
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grads

    x = np.random.rand(100, 784)

#ミニバッチ学習の実装
import numpy as np
from dateset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \load_mnist(nomalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
#ハイパーパラメーター
network = TwoLayerNet(input_size=784, hidden_size=50, out_size=10)

for i in range(iters_num):
#ミニバッチの取得
  batch_mask = np.random.shoice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  #勾配の計算
  grad = networl.numerical_gradient(x_batch, t_batch)

  #パラメーターの更新
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.prams[key] -= learning_rate * grad[key]

  #学習経過の記録
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

#テストデータ
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train),(x_test, t_test) = \load_mnist(normalize = True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

iters_num = 10000
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

grad = neteork.numerical_gradient(x_batch, t_batch)
for key in ('W1', 'b1', 'W2', 'b2'):
  network.prams[key] -= learning_rate * grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    prin("train_acc, test_acc | "+ str(train_acc)+ ", " + str(test_acc))

#乗算レイヤ
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x = y

    return out

  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x#xとyをひっくり返す
    return dx, dy
#加算レイヤ
class AddLayer:
  def __init__ (self):
    pass
  def forward(self, x, y):
    out = x + y
    return out
  def backard(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy
#sigmoid layer
class Sigmoid:
  def __init__(self):
    self.out = None

  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out

  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out

    return dx

#Affine/Softmax layer
class Affine:
  def __init__ (self, w, b):
    self.w = w
    self.b = b
    self.x = None
    self.dw = None
    self.db = None
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out
  def backward(self, dout):
    dx = np.dot(x, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx

#Softmax-with-Loss Layer
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.loss = cross_entropy_error(self.y, self.t)

    return self.loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx

    #実装
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

  def __init__ (self, input_size, hiden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * \np.random.random.randn(input_*size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * \ np.random.randn(hidden_size, output_size)
    self.params['b2'] = np,zeros(output_size)
    #create layer
    self.layers = orderedDict()
    self.layers ['Affinel'] = \ Affine(self.params['b1'])
    self.layers ['Affine2'] = \ Affine(self.params['b2'])
    self.lastLayer = SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layer.values():
      x = layer.forward(x)
    return x
    # x:入力 y:教師
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim ! = 1 : t = nnp.argmax(t, axis=1)

    accourcy = np.sum(y == t) / float(x, shape[0])
    return accuracy

    def numerical_gradient(self, x, t):
      loss_w = lambda W: self.loss(x, t)

      grads={}
      grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
      grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
      grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
      grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads

    def gradient(self, x, t):
      self.loss(x, t)#forward
      dout = 1#backward
      dout = self.lastlayer.backward(dout)

      layers = list(self.layers.valus())
      layers.reverse()
      for layer in layers:
        dout = layer.backward(dout)

      grads = {}
      grads['W1'] = self.layers['Affine1'].dW 
      grads['b1'] = self.layers['Affine1'].db 
      grads['W2'] = self.layers['Affine2'].dW 
      grads['b2'] = self.layers['Affine2'].db

      return grads

#誤差逆伝播法の勾配確認
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dateset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test)= \ load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradien(x_batch, t_batch)

#各重みの絶対時差の平均を求める
for key in grad_numerical.keys():
  diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ":" + str(diff))

#誤差逆伝搬法を使った学習
import sys, os
sys.path.append(os.perdir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwolayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = \ load_mnist(normalize = True)

network = TwolayerNet(input_size=784, hidden_size=50, output_seze=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list =[]

iter_per_epoch = max(train_size) / batch_size, 1)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch.mask]

  #誤差逆伝搬法によって勾配を求める
  grad = network.gradient(x_batch, t_batch)

  for key in ('W1', 'b1' 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % inter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_loss_list.append(loss)

  if i % iter_per_epoch ==0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_ass_list.append(test_acc)
    print(train_acc, test_acc)

#SGD
class SGD:
  def __init__ (self, lr=0.01):
    self.lr = lr
  def update (self, prams, grads):
    for key in params.keys():
      params[key] -= self.lr * grads[key]

#Momentum
class Momentum:
  def __init__ (self, lr=0.01, momentum=0.9):
    self.lr = lr
    self.momentum = momentum
    self.v = None
  def update(self, params, grads):
    if self.v ={}
    for key, val in params.items():
      self.v[key]=np.zeros_like(val)

    for key in params.keys():
      self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
      params[key] += self.v[key]

      #AdaGrad
class AdaGrad:
  def __init__(self, lr=0.01):
    self.lr =lr
    self.h = None:
    self.h = {}
    for key, val in params.items():
      self.h[key] += grads[key] * grads[key]
      params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + le-7))

    #隠れ層のアクティベーション分布
    import numpy as np
    import matplotlib.pyplot as plt

    def sigmoid(x):
      return 1/ (1+np.exp(-x))

    x = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    for i in range(hidden_layer_size):
      if i != 0:
        x = activations[i - 1]

      w = np.random.randn(node_num, node_num) *1

      z = np.dot(x,w)
      a = sigmoid(z)
      activations[i] = a


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01) # 学習係数 0.01 の SGD でパラメータ更新
     max_epochs = 201
     train_size = x_train.shape[0]
     batch_size = 100

     train_loss_list = []
     train_acc_list = []
     test_acc_list = []
     iter_per_epoch = max(train_size / batch_size, 1)
     epoch_cnt = 0

     for i in range(1000000000):
         batch_mask = np.random.choice(train_size, batch_size)
         x_batch = x_train[batch_mask]
         t_batch = t_train[batch_mask]
         grads = network.gradient(x_batch, t_batch)
         optimizer.update(network.params, grads)

if i % iter_per_epoch == 0:
             train_acc = network.accuracy(x_train, t_train)
             test_acc = network.accuracy(x_test, t_test)
             train_acc_list.append(train_acc)
             test_acc_list.append(test_acc)
             epoch_cnt += 1
             if epoch_cnt >= max_epochs:
break

#Relu layer
class Relu:
  def __init__(self, x):
    seof.mask = None
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] =0

    return out

  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout

    return dx

    #sigmoid layer
class Sigmoid:
  def __init__(self):
    self.out = None

  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

  def backward(self, dout):
    dx = dout * (1.0 - self.out) = self.out

    return dx

#soft max with loss
class softmaxWithLoss:
  def __init__(self):
    self.loss = None #損失
    self.y = None#softmaxの出力
    self.t = None#教師データ(one_hot vector)

  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    return self.loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx
#ニューラルねとワークの全体像
##前提　重みとバイアスを訓練データに適応するように調整することを学習を呼ぶ
##1.ミニバッチ訓練データの中からランダムに一部のデータを選び出す
##2.勾配の算出各重みパラメータに関する損失関数の勾配を求める
##3.パラメータの更新重みパラメータを勾配方向に微小量だけ更新する
##4.繰り返し