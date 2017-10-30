import sys, os
sys.path.append(os, pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    #重みの初期化
    self.params ={}#size=ニューロンの数
    self.params['W1']= weight_init_std * \
                       np.random.randn(input_size, hidden_size)
    self.params['b1']=np.zeros(hidden_size)
    self.params['W2']=weight_init_std * \
                       np.random.randn(hidden_size, output_size)
    self.params['b2']=np.zeros(output_size)

  def predict(self, x):#推論（認識）を行う
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y

  def loss(self, x, t):#損失関数（交差エントロピー誤差）
    y= self.predict(x)

    return cross_entropy_error(y, t)

  def accuracy(self, x, t): #精度を確認
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(y, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lamdba W: self.loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])#numerical_gradient=重みパラメータについて勾配を求める
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads