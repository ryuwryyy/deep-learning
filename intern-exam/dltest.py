def mean_squared_error(t, y):
    return 0.5*np.sum((y-t))**2)

def sigmoid(x):
    beta = 10
    return 1/ (1 + np.exp(-x*beta)

# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)

#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + h
#         fxh1 = f(x)
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 - fxh2)/(2*h)

#         x[idx] = tmp_val
#         it.iternext()

#     return grad

x = [[0,0],[0,1],[1,0],[1,1]]
t=[0,0,0,1]

class ThreeNodeNet:
    def __init__(self)
        self.params={}
        self.params['W1'] = np.array(np.random.rand(3), np.random.rand(3))
        self.params['b1'] = np.random.rand(3)*-1
        self.params['W2'] = np.random.rand(3)
        self.params['b2'] = np.random.rand(1)
    def predict(self, x):
        A1 = np.dot(x, self.params['W1']) + self.params['b1']
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, self.params['W2']) + self.params['b2']
        Z2 = sigmoid(A2)
        return Z2
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(t, y)
    def numerical_gradient():
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1']=numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

net2 = ThreeNodeNet()
net2.preredict(x)

learning_rate2=0.05
train_loss_list2=[]

for i in range(10000):
    grad = net2.numerical_gradient(x,t)
    for key in ('W1', 'b1', 'W2', 'b2'):
        net2.params[key] -= learning_rate2 * grad[key]
        loss = net2.loss(x,t)
        train_loss_list2.append(loss)