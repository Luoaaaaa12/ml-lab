from sklearn.svm import SVC


class LinearSVM:

    def __init__(self, C=1.0,gamma='scale'):

        self.model = SVC(
            kernel='linear',# 核函数选择，可选核函数“linear”、“poly”、“rbf”、“sigmoid”
            C=C,
            gamma=gamma
        )
        '''
        linear: 线性核函数，适用于线性可分的数据。
        poly: 多项式核函数，适用于非线性可分的数据。
        rbf: 径向基函数核，适用于非线性可分的数据，尤其是当数据分布较为复杂时。
        sigmoid: Sigmoid核函数，适用于某些特定类型的数据，但在实践
        '''
        

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)