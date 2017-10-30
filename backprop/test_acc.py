import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayereNet

(x_train, t?train), (x_test, t_test)= \
    load_mnist(nomaloze=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list =[]
#1エボックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

#ハイパーパラメータ
iters_num = 10000
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet = np.random.choice(input_size =784, hidden_size=50, output_size=10)

for i in range(iters_num):
  #ミニバッヂの取得
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t.batch = t_train[batch_mask]

  #勾配の算出
  grad = network.numerical.gradient(x_batch, t_batch)

  #パラメ-たの更新（重さ）
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss.list.append(loss)

  #1エボックごとににんしき精度を算定 絵ポッ苦とは訓練データをすべて使い切ったときの回数に対応
  if i % iter_per_eposh == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc |" + str(train_acc) + "," + str(test_acc))