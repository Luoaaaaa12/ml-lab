import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)  # 初始化超平面
        self.bias = 0

        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias

                y_predicted = np.sign(linear_output)

                if y_predicted != y[idx]:

                    self.weights += self.lr * y[idx] * x_i
                    self.bias += self.lr * y[idx]

    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias

        return np.sign(linear_output)