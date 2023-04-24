from NeuralNetworks import Perceptron
import numpy as np


def main():
    test = Perceptron()
    print(test(np.ones(10)))


if __name__ == '__main__':
    main()
