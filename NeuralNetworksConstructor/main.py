import numpy as np
from Layers import RadialBasisLayer, LinearLayer
from NeuralNetworks import RBNN


def main():
    x = np.array([1e-6, 1e-6])
    l1 = RadialBasisLayer(2, 10)
    l2 = LinearLayer(10, 1)
    print(l1.weights.shape)
    print(l1.bias.shape)
    print(l1(x))
    print(
        l1.gradient(np.array([[0.1]]), l2).shape
    )
    print(l1.backward(np.arange(10).reshape([1, -1]), x))

    # test = RBNN()
    # res = test(np.array([0.4, 0.5]))
    # print(res)


if __name__ == '__main__':
    main()
