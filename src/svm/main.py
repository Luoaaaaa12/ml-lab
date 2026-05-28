from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(
    n_samples=200,
    noise=0.2,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.show()