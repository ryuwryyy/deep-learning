def back_propagation(x, z1, y, d):
    global W1
    global W2
    delta2 = y - d
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)

    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W