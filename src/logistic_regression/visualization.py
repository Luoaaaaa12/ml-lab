import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(X, y, model):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)

    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.3)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors='k'
    )

    plt.savefig("results/logistic_boundary.png")
def plot_loss(losses):

    plt.plot(losses)

    plt.title("Loss Curve")

    plt.xlabel("Iterations")

    plt.ylabel("Loss")

    plt.savefig("results/logistic_loss_curve.png")
    plt.show()