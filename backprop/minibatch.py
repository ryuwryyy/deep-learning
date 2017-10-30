#1.ミニバッチ（一部のデータをランダムに ミニバッチの「損失関数の値を減らすこと」を目的とする
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_nist(normalize=True, one_hot_label=True)#normalize=入力画像を0.0〜1.0の値に正規化するかどうかの設定
    #one_hot_label=正解が1、他は0にするone_hot表現として格納するかどうかの設定

train_loss_list = []

#ハイパーパラメータ
iters_num = 10000#iteration(繰り返し)の数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output=10)

for i in range(iters_num):
  #ミニバッチの取得
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  #勾配の計算
  grad = network.numrival_gradient(x_batch, t_batch)

  #重さパラメータの更新
  for key in ('W1, 'b1', 'W2', 'b2'):
    network.params[key] -= learng_rate * grad[key] #paramsとは

    #学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)#末尾に追加