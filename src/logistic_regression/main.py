from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from logistic_regression import LogisticRegression
from visualization import plot_loss, plot_decision_boundary
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(
    learning_rate=0.01,
    n_iters=1000
)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
plot_loss(clf.losses)
plot_decision_boundary(X_test, y_test, clf)
print("Accuracy:", accuracy)