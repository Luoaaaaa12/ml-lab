from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from softmax_regression import SoftmaxRegression
from visualization import plot_loss

X, y = make_blobs(
    n_samples=1000,
    centers=3,
    n_features=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

clf = SoftmaxRegression(
    learning_rate=0.01,
    n_iters=1000
)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
plot_loss(clf.losses)
print("Accuracy:", accuracy)