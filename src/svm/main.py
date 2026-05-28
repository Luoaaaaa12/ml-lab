from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from perceptron import Perceptron
from visualization import plot_decision_boundary

X, y = make_moons(
    n_samples=200,
    noise=0.2,
    random_state=42
)

y = 2 * y - 1

clf = Perceptron()

clf.fit(X, y)

plot_decision_boundary(X, y, clf)
