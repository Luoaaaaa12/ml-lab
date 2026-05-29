import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):

        self.lr = learning_rate
        self.n_iters = n_iters

        self.weights = None
        self.bias = None
        
        self.losses = []

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):

            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self.sigmoid(linear_model)
            
            loss = -np.mean(
                y*np.log(y_predicted) + 
                (1 - y)*np.log(1 - y_predicted+1e-15)
            )
            self.losses.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        y_predicted = self.sigmoid(linear_model)

        predicted_labels = [1 if i > 0.5 else 0 for i in y_predicted]

        return np.array(predicted_labels)