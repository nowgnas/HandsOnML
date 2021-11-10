from sklearn.datasets import fetch_openml
import numpy as np


def get_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    return mnist


if __name__ == '__main__':
    mnist = get_mnist()
    print(type(mnist))
    print(mnist.keys())
