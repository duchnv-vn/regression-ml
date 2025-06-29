import numpy as np


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Applying Softmax Activation function
def softmax(logits):
    exponential = np.exp(logits)
    return exponential / np.sum(exponential, axis=1).reshape(-1, 1)


# calculating gradient of logistic regression
def sigmoid_gradient(sigmoid):
    return np.multiply(sigmoid, (1 - sigmoid))


# Computing Loss over using logistic regression
def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]


# Adding bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


# Basically doing prediction but named forward as its
# performing Forward-Propagation
def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat, h


# performing back-Propagation
def back(X, Y, y_hat, w2, h):
    w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
    w1_gradient = np.matmul(prepend_bias(X).T, np.matmul(y_hat - Y, w2[1:].T)
                            * sigmoid_gradient(h)) / X.shape[0]
    return w1_gradient, w2_gradient


# Calling the predict() function
def classify(X, w1, w2):
    y_hat, _ = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


# initializing weights for all nodes
def initialize_weights(n_input_variables, n_hidden_nodes, n_classes):
    w1_rows = n_input_variables + 1
    w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)

    return w1, w2


def prepare_batches(X_train, Y_train, batch_size):
    x_batches = []
    y_batches = []
    n_examples = X_train.shape[0]
    for batch in range(0, n_examples, batch_size):
        batch_end = batch + batch_size
        x_batches.append(X_train[batch:batch_end])
        y_batches.append(Y_train[batch:batch_end])
    return x_batches, y_batches


# Printing details of each iteration on console
def report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat, _ = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print("%5d-%d > Loss: %.8f, Accuracy: %.2f%%" %
          (epoch, batch, training_loss, accuracy))


# Training phase for neural network
def train(X_train, Y_train, X_test, Y_test, n_hidden_nodes,
          epochs, batch_size, lr):
    n_input_variables = X_train.shape[1]
    n_classes = Y_train.shape[1]
    # Initializing random weights
    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    x_batches, y_batches = prepare_batches(X_train, Y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(len(x_batches)):
            y_hat, h = forward(x_batches[batch], w1, w2)
            w1_gradient, w2_gradient = back(x_batches[batch], y_batches[batch],
                                            y_hat, w2, h)
            w1 = w1 - (w1_gradient * lr)
            w2 = w2 - (w2_gradient * lr)
            report(epoch, batch, X_train, Y_train, X_test, Y_test, w1, w2)
    return (w1, w2)
