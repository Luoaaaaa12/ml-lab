import matplotlib.pyplot as plt


def plot_loss(losses):

    plt.plot(losses)

    plt.title("Softmax Loss Curve")

    plt.xlabel("Iterations")

    plt.ylabel("Loss")

    plt.show()