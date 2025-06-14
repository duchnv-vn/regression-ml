import numpy as np


def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(logit):
    exponential = np.exp(logit)
    # max_expo = np.max(exponential)
    return exponential / np.sum(exponential, axis=1).reshape(-1, 1)


def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    return softmax(np.matmul(prepend_bias(h), w2))


def cross_entropy_loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]


def classify(X, w1, w2):
    y_hat = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)


def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat = forward(X_train, w1, w2)
    training_loss = cross_entropy_loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2f%%" %
          (iteration, training_loss, accuracy))


def train():
    pass
