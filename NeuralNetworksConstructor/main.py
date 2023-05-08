from Architectures import Perceptron
import numpy as np


def main():
    test = Perceptron()
    with test:
        print(test)
        print(test(np.ones(2)))


if __name__ == '__main__':
    main()

