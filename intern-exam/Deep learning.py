# パーセプトロン AND
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# 重みとバイアス
import numpy as np

x = np.array([0, 1]) # 入力
w = np.array([0.5, 0.5]) # 重み
b = -0.7 # バイアス
x*w
np.sum(w*x)
np.sum(w*x) + b

# 重みとバイアスとAND
def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, .0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1
#NAND
def NAND(x1, x2)
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5]) #重みがANDと違う
  b = 0.7
  tmp = np.sum(x*w) + b
  if tmp <= 0:
    return 0
  else
    return 1
#OR
def OR(x1, x2)
  x=np.array([x1, x2])
  y=np.array([0.5, 0.5]) # 重みとバイアスがANDと違う
  b=-0.2
  tmp=np.array(x*w) + b
  if tmp <= 0:
    return 0
  else:
    return 1

#XORゲート
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

#ステップ関数
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0
# 引数のxは実数（浮動小数点数）しか入力できない np.array()が使えない
def step_function(x):
  y = x > 0
  return y.astype(np.int)

#Numpyの中身
import numpy as np ¥
x = np.array([-1.0, -1.0, 2.0])
x # >>> array([-1, 1, 2,0])
y = x > 0
# y #>>> array ([False, True, True]), dtype=bool
y = y.astype(np.int)# 小数点

x = np

#ジグモイド関数の実装
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.ylim(-0.1, 1.1)
plt.show()

#RuLU関数
def relu(x):
  return np.maximum(0, x)

#多次元配列
import numpy as np
A = np.array([1,2,3,4])
B = np.array([[5,6],[7,8])
np.ndim(A) #行
A.shape #列
np.dot(A, B)#積

#ニューラルネットワークの行列の積
X = np,array([1, 2])
W = np,array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2)+B2
Z2 = sigmid(A2)

def indentity_function(x): #恒等関数
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function()

#まとめ
def init_network(): #重みとバイアスの初期化と格納
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])

  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def forward(network, x): #入力から出力への変換
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  zl = sigmoid(a1)

  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_funtion(a3)

  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708  0.69627909]

  #ソフトマックス関数
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 指数関数
print(exp_a)#>>>[ 1.34985881 18.17414537 54.59815003]

sum_exp_a = np.sum(exp_a)#指数関数の和

def softmax(a):#ソフトまくっす関数の定義
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp / sum_exp_a

  return y

#オーバーフロー対策
def softmax(a)
  c = np.max(a)
  exp_a = np.exp(a-c)#ここ注意
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

#2乗和誤差
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def mean_squared_error(y, t):
  return 0.5 * np.sum((y - t)**2)
  t = [0,0,1,0,0,0,0,0,0,0]#『2』を正解とする
  mean_squared_error(np.array(t))
  y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
  mean_squared_error(np.array(y), np.array(t))

#交差エントロピー誤差
def cross_entrory_error(y, t):
  delta = le - 7
  return np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))

#ミニバッチ学習
import sys, os
sys.path.apend(os.pardir)
import numpy as np
from dateset.mlist import load_mnist

(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.shoice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

np.random.choice(60000, 10)

#交差エントロピー誤差の実装
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  batch_size = y.shape[0]
  return np.sum(t * np.lof(y)) / batch_size)

#バッチ対応版交差エントロピー誤差

def cross_entropy.error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batxh_size = y.shape[0]
  return np.sum(t * np.log(y)) / batch.size

#微分の実装
def function_1(x):
  return 0.01*x**2 + 0.1*x

  import numpy as np
  import matplotlib.pylab as plt

  x = np.arange(0.0, 20,0, 0.1) # 0 から 20 まで、0.1 刻みの x 配列
  y = function_1(x)
  plt.xlabl("f(x)")
  plt.plot(x, y)
  plt.show()

#偏微分
def function_tmp1(x0):
  return x0*x0 + 4.0**2.0
  numerical_diff(function_tmp2, 4.0)
#勾配
def numerical_gradient(f, x):
  h = le.4 #0.0001
  grad = np.zeros_like(x) #xと同じ形状の破裂を生成

  for idx in range(x, size):
    tmp.val = x[isx]#f【x＋h】の計算
    x[idx] = tmp.val - h
    fxh1 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp.val #値を元に戻す

  return grad

#勾配法
def gradient_descent(f, init_x, lr=0.01, step.num=100):
  x = init_x

  for i in range(step.num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

#学習あるごリズモの実装
###1.ミニバッチ
###2.勾配の算出
###3.パラメーターの更新
###4.繰り返し

import sys, os
sys.path.append(os, pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwolayerNet:

  def __init__(self, input.size, hidden_size, output_size, weight_init_std=0.01):
    self.prams ={}
    self.prams =["w1"] = weight_init_std * \ np.random.randn8input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * \ np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
  def predict(self, x):
    W1, W2 = self.prams['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y

  def loss(self, x, t):
    y = self.predict(x)

    return cross_entropy_error(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)

    gradas = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grads

#誤差逆伝搬法
##乗算レイヤ
class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x *y

    return out

  def backward(self, dout)*
  dx = dout * self.y
  dy = dout * self.x

  return dx, dy
##加算レイヤ
class Addlayer:
  def __init__(self):
    pass
  def forward(self, x, y):
    out = x +y
    return out

  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy