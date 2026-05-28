from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from perceptron import Perceptron
from visualization import plot_decision_boundary
from svm_sklearn import LinearSVM

X, y = make_moons(
    n_samples=200,
    noise=0.2,
    random_state=42
)

y = 2 * y - 1

clf = LinearSVM(C=1)

clf.fit(X, y)

plot_decision_boundary(X, y, clf , save_path='results/svm_decision_boundary.png')
