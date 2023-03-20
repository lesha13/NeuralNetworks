import numpy as np

from NeuralNetworks import NeuralNetwork
from Layers import RadialBasisLayer, LinearLayer
from loss_functions import CrossEntropyLoss


class RBNN(NeuralNetwork):
    def __init__(self):
        self.radial_basis_layer = RadialBasisLayer(2, 10)
        self.linear_layer = LinearLayer(10, 1)

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1) -> None:
        loss = CrossEntropyLoss(self(x), y)

        error = self.linear_layer.gradient(loss)
        self.linear_layer.backward(error, self.radial_basis_layer, lr)
