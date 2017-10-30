import sys, os
sys.path.append(os, pardier)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
  def __init__(self, input_size, output_size,weight_init?std=0.01):
      #重みの初期化
      self.params = {}
      self.params['W1']=weight_init_std \ np.random.randn(input_size, hidden_size)
      self.params['b1']=np.zeros(hidden_seize)
      self.params['W2']=weight_init_std * \ nip.random.randn(hidden_size, output_size)

      self.params['b2'] = np.zeros(output_size)
      #レイヤーの生成
      self.layers = OrderedDict()
      self.layers['Affinel'] = \ Affine(self.params['W1'], self.params['b1'])
      self.layers['Rule1']= Rule()
      self.layers['Affinel2'] = \ Affine(self.params['W2'], self.params['b2'])
      self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
      for layer in self.layers.values():
        x=layer.forward(x)

      return x

#x:入力データt:教師データ
def numerical_gradient(self, x, t):
  loss_W = lambda W: self.loss(x, t)
  return self.lastLayer.forward(y, t)

def accuracy(self, x, t):
  y = self.predict(x)
  y = np.argmax(y, axis = 1)
  if t.ndim !=1:t = np.argmax(t, axis=1)

  accuracy = np.sum(y == t) / float(x, shape[0])
  return accuracy

#x:入力データ t＝教師データ
def numerical_gradient(self, x, t):
  loss_W = lambda W: self.loss(x, t)

  grads = {}
  grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
  grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
  grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
  grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
  return grads

def gradient(self, x, t):
  #forward
  self.loss(x, t)

  #backward
  dout= 1
  dout = self.lastLayer.backward(dout)

  layers = list(self.layers.values())
  laers.reverse()
  for layer in layers:
    dout = layer.backward(dout)
  #設定
  grads = {}
  grads['W1'] = self.layers['Affine1'].dW 
  grads['b1'] = self.layers['Affine1'].db 
  grads['W2'] = self.layers['Affine2'].dW 
  grads['b2'] = self.layers['Affine2'].db

  return grads

  import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderdDict

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, 
                 weight_init_std = 0.01):
        
        self.params = {}
        self.params['W1']= weight_init_std *　\　#std　標準偏差値
                                          np.random.randn(input_size, hidden_size)
        self.params[ 'b1'] = np.zeros(hidden_size)
        self.params['W2'] =weight_init_std *　\
                                          np.random.randn(hidden_size, output_size)
        self.params[ 'b2'] =np.zeros(output_size)
        
        #　レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affinel'] = \
                Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
                Affine(self.params['W2'], self.params['b2'])
            
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.value():
            x = layer.forward(x)
            
        return x
    
    #x：入力データ,　ｔ：教師データ
    def loss(self, x, t):
        y = self.predict(x)
    return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndm ! = 1 : t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numrical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    defgradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backwdard(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW grads['b1'] = self.layers['Affine1'].db grads['W2'] = self.layers['Affine2'].dW grads['b2'] = self.layers['Affine2'].db
        return grads

#誤差逆伝搬法を使った学習
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(nomalize=True, one_hot_label=True)

network = TwolayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list= []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  #誤差逆伝搬法によって勾配を求める
  grad = network.gradient(x_batch, t_batch)

  #更新
  for key in ('W1','b1', 'W2', 'b2'):
    network.params[key] -= learning_rate = grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(train_acc test_acc)

#復習 3層ニューラルネットワーク
def init_network():
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3)

  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

#微分の復習 中心差分
def numrical_diff(f, x):
  h = 1e - 4
  return (f(x+h) - f(x-h) / (2*h))
#数値微分
  def function_1(x):
    return 0.01*x**2 + 0.1*x
#偏微分
def function_2(x):
  return x[0]**2 + x[1]**2
#勾配
def numerival_gradient(f, x):
  h = 1e - 4
  grad = np.zeros_like(x)

  for idx in range(x.size): #.size=要素数
  tmp_val = x[idx]

  x[idx] = tmp_val + h #f（x+h）の計算
  fxh1 = f(x)

  x[idx] = tmp_val - h
  fxh2 = f(x)

  grad[idx] = (fxh1 -fxh2) / (2*h)
  x[idx] = tmp_val #値を元に戻す

return grad

#勾配法
def gradient_descent(f, init_x, lr=0.01, step_num=100):#f=最適化したい関数 lr=learning rate(学習率) step_num=勾配法による繰り返しの数
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad #勾配（numerical_gradient(f, x）に学習率をかける

return x

#ニュラルネットワックに対する勾配
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
  def __inif__(self):
    self.W = np.random.randn(2, 3)#ガウス分布で初期化 W=重み

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x): #損失関数
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss
    #argmax (argument of the max)で最大集合点（定義域の集合）
    #lambda 簡単な関数ならlambda
    f = lambda w: net.loss(x, t)
    dw = numerical_gradient(f, net.W)

#学習を実装
#0.学習とは、重みとバイアスを訓練データに適応するように調整すること

#1.ミニバッチ（一部のデータをランダムに ミニバッチの「損失関数の値を減らすこと」を目的とする

#2.勾配の算出（重みの勾配を求める 損失関数を減らすため）

#3.重みパラメーターの更新（勾配”方向”に微小量、更新）
#4.繰り返し