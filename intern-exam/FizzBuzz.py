def mean_sequared_error(y, t):
  return 0.5*(y-t)**2
def sigmoid(x):
  beta = 10
  return 1 /(1 + np.exp(-x*beta))

  class ThreeNodeNet():
    def __initt__(self):
      self.params={}
      self.params['W1'] = np.array(np.random.rand(3), np.random.rand(3))
      self.params['b1'] = np.random.rand(3)
      self.params['W2'] = np.random.rand(3)
      self.params['b2'] = np.random.rand(1)

    def predect(x, t):
      A1 = np.dot(x, self.params['W1']) + self.params['b1']
      Z1 = sigmoid(A1)
      A2 = np.dot(Z1, self.params['W2']) + self.params['b2']
      Z2 = sigmoid(A2)

    def loss(x, t):
      y = self.predict(x,t)
      return mean_sequared_error(y, t)