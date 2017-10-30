#XOR
  def XOR(x1, x2):
    s1 = NAND(x1. x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
  return y

#1.ニューラルネットワーク（誤差逆伝搬法）

##import
  import sys, os
  sys.path.append(os.pardir)
  import numpy as np
  from common layers import *
  from common.gradient import numerical_gradient
  from collections import OrderedDict

##TowLayerNet
  class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
###params=ディクショナリー変数
      self.params = {}
      self.params['W1'] = weight_init_std * \
                          np.random.randn(input_size, hidedn_size)
      self.params['b1'] = np.zeros(hidden_size)
      self,params['W2'] = weight_init_std * \
                          np.random.randn(hidden_size, output_size)
      self.params['b2'] = np.zeros(output_size)
  ###レイヤの生成 layers=順番付きディクショナリ
      self.layers = OrderedDict()
      self.layers['Affine1'] = \
        Affine(self.params['W1'], self.params['b1'])
      self.layers['Relu1'] = Relu()
      self.layers['Affine2'] = \
        Affine(self.params['W2'], self.params['b2'])

      self.lastLayer = SoftmaxWithLoss()

##認識 推論を行う
    def predict(self, x):
      for layer in self.layers.values():
        x = layer.forward(x)
      return x
  ###損失関数 t=正解ラベル
    def loss(self, x, t):
      y = self.predict(x)
      return self.lastLayer.forward(y, t)
  ###認識精度を求める
    def accuracy(self, x, t):
      y = self.predict(x)
      y = np.argmax(t, axis=1)
      if t.ndim != 1 : t = np.argmax(t, axis=1)#.ndim=次元数

      accuracy = np.sum(y == t)  / float(x.shape[0])#浮動小数点を表すfloat
      return accuracy
###重みに勾配を数値微分によって求める
    def numerical_gradient(self, x, t):
      loss_W = lambda W:self.loss(x, t)

      grads = {}
      grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
      grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
      grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
      grads['b2'] = numerical_gradient(losss_W, self.params['b2'])
      return grads
###重みの勾配を誤差逆伝播法
    def gradient(self, x, t):
      self.loss(x, t)

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



#2.勾配確認（数値微分と誤差逆伝搬法の結果を比較して正しさを図る）

##import
  import sys, os
  sys.path.append(os, pardir)
  import numpy as np
  from dataset.mnist import load_mnist
  from two_layer_net import TwoLayerNet

##データの読み込み
  (x_train, t_train), (x_test, t_test) = \
      load_mnist(normalize=True, one_hot_label=True)

  network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

  x_batch = x_train[:3]
  t_batch = t_train[:3]

  grad_numerical = network.numerical_gradient(x_batch, t_batch)
  grad_backprop = network.gradient(x_batch, t_batch)

##各重みの絶対誤差の平均を求める
  for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))



#3.誤差逆伝搬法を使った学習

##import
  import sys, os
  sys.path.append(os, pardir)
  import numpy as np
  from dataset.mnist import load_mnist
  from two_layer_net import TwoLayerNet

##データの読み込み
  (x_train, t_train), (x_test, t_test) = \
      load_mnist(nomalize=True, one_hot_label=True)
      #nomalize=0.0〜1.0の値に正規化するかどうかを設定
      # onehotlabel =正解となるラベルだけが1であとが全部0の配列

  network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

  iters_num = 10000
  train_size = x_train.shape[0]
  batch_size = 100
  learning_rate = 0.1

  train_loss_list = []
  train_acc_list = [] #acc = adaptive  cruise control
  test_acc_list = []

  iter_per_epoch = max(train_size / batch_size, 1) #訓練データをすべて使い切ったときの回数

  for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

###誤差逆伝搬法によって勾配を求める
    grad = network.gradient(x_batch, t_batch)

  ##更新
    for key in ('W1', 'b1', 'W2', 'b2'):
      network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
      train_acc = network.accuracy(x_train, t_train)
      test_acc = network.accuracy(x_test, t_test)
      train_acc_list.append(train_acc)
      test_acc_list.append(test_acc)
      print(train_acc, test_acc)