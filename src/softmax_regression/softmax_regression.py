import numpy as np


class SoftmaxRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):

        self.lr = learning_rate
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

        self.losses = []

    def softmax(self, z):

        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):

        one_hot_y = np.zeros((len(y), num_classes))

        one_hot_y[np.arange(len(y)), y] = 1

        return one_hot_y

    def fit(self, X, y):

        n_samples, n_features = X.shape

        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))

        self.bias = np.zeros((1, n_classes))

        y_one_hot = self.one_hot(y, n_classes)

        for _ in range(self.n_iters):

            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self.softmax(linear_model)

            loss = -np.mean(
                np.sum(
                    y_one_hot * np.log(y_predicted + 1e-9),
                    axis=1
                )
            )

            self.losses.append(loss)

            dw = (1 / n_samples) * np.dot(
                X.T,
                (y_predicted - y_one_hot)
            )

            db = (1 / n_samples) * np.sum(
                y_predicted - y_one_hot,
                axis=0,
                keepdims=True
            )

            self.weights -= self.lr * dw

            self.bias -= self.lr * db

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        y_predicted = self.softmax(linear_model)

        return np.argmax(y_predicted, axis=1)