x=[[0,0],[0,1],[1,0],[1,1]]
t=[0,1,1,0]

def mean_sequared_error(y, t):
  return 0.5 * np.sum((y-t)**2)

def sigmoid(x):
  beta=10
  return 1 / (1 + np.exp(-x * beta))

class ThreeNodeNet():
  def __init__(self):
    self.params={}
    self.prams['W1']=np.array(np.random.rand(3), np.random.rand(3))
    self.params['b1'] = np.random.rand(3)
    self.params['W2'] = np.random.rand(3)
    self.params['b2'] = np.random.rand(1)

  def predict(x, t):
    A1 = np.dot(x, self.params['W1']) + self.params['b1']
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, self.params['W2']) + self.params['b2']
    Z2 = sigmoid(A2)

  return Z2

  def loss(self,x, t):
    y = self.predict(x)
    return mean_sequared_error(y,t)

  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x,t)
    grad = {}
    grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grads

net2=ThreeNodeNet()
net2.predict(x, t)

learning_rate = 0.05
train_loss_list=[]

for i in range(10000):
  grad = net2.numerical_gradient(x, t)
  for key in ['W1', 'b1', 'W2', 'b2']:
    net2.params[key] -= learning_rate * grad[key]
    loss = net2.loss(x, t)
    train_loss_list.append(loss)