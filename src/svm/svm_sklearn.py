from sklearn.svm import SVC


class LinearSVM:

    def __init__(self, C=1.0):

        self.model = SVC(
            kernel='linear',
            C=C
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)