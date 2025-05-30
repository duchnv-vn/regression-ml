import numpy as np


# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Performing Forward-Propagation
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)


def classify(X, w):
    return np.round(forward(X, w))


# Computing Loss using log loss
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


# Calculating gradient
def gradient(X, Y, w):
    return (np.matmul(X.T, (forward(X, w) - Y))) / X.shape[0]


def train(X, Y, iter, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iter):
        print('Iteration %4d => Loss: %.20f' % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))
