
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class SigmoidLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            W = rng.uniform(low=-4*np.sqrt(6. / (n_in + n_out)),
                            high=4*np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out))
        if b is None:
            b = np.zeros(n_out)

        self.W = W
        self.b = b

    def forward(self, input=None):
        if input is not None:
            self.input = input
        lin_output = np.dot(self.input, self.W) + self.b
        self.output = sigmoid(lin_output)
        return self.output

    def backward(self, d, lr):
        d_next = np.dot(d, self.W.T) * self.input * (1-self.input)
        for i in xrange(self.input.shape[0]):
            xi = self.input[i:i+1].T
            di = d[i:i+1]
            self.W -= lr * np.dot(xi, di)
            self.b -= lr * d[i]
        return d_next

    def get_outputlayer_delta(self, y):
        return (self.output-y) * self.output * (1-self.output)


class MLP(object):
    def __init__(self, rng, input, label, n_in, hidden_layer_sizes, n_out):
        self.rng = rng
        self.input = input
        self.label = label

        self.hidden_layers = []
        n_layers = len(hidden_layer_sizes)
        self.n_layers = n_layers

        for i in range(n_layers):
            if i == 0:
                input_size = n_in
                layer_input = self.input
            else:
                input_size = hidden_layer_sizes[i-1]
                layer_input = self.hidden_layers[i-1].forward()

            self.hidden_layers.append(
                SigmoidLayer(rng, layer_input, input_size,
                             hidden_layer_sizes[i])
            )
        self.output_layer = SigmoidLayer(
                                rng,
                                self.hidden_layers[n_layers-1].forward(),
                                hidden_layer_sizes[n_layers-1],
                                n_out
                            )

    def predict(self, x=None):
        if x is None:
            x = self.input

        for i in range(self.n_layers):
            x = self.hidden_layers[i].forward(x)
        return self.output_layer.forward(x)

    def train(self, lr=0.1, epochs=1000, batch_size=5):
        X = self.input
        t = self.label

        for i in xrange(epochs):
            index = self.rng.permutation(X.shape[0])
            for n in range(0, X.shape[0], batch_size):
                g = X[index[n:n+batch_size]]
                for j in range(self.n_layers):
                    g = self.hidden_layers[j].forward(g)
                g = self.output_layer.forward(g)

                d = self.output_layer.get_outputlayer_delta(t[index[n:n+batch_size]])
                d = self.output_layer.backward(d, lr)
                for j in range(self.n_layers)[::-1]:
                    d = self.hidden_layers[j].backward(d, lr)

            loss_sum = np.sum((self.predict() - t)**2)
            print("epoch: {0:5d}, loss: {1:.5f}".format(i, loss_sum))

if __name__ == '__main__':
    m = 50
    x_train = np.linspace(-1, 1, m).reshape((m, 1))
    y_train = np.abs(x_train)
    rng = np.random.RandomState(123)
    mlp = MLP(rng, x_train, y_train, len(x_train[0]),
              [4, 4], len(y_train[0]))
    mlp.train()
    y_hat = mlp.predict(x_train)
    plt.scatter(x_train, y_train, color='r')
    plt.scatter(x_train, y_hat, color='b')
    plt.savefig("mlp_approximate_abs_10000.png")
    plt.show()