from NeuralNetworks import NeuralNetwork
from Layers import RadialBasisLayer, DenseLayer


class RBNN(NeuralNetwork):
    def __init__(self):
        self.radial_basis_layer = RadialBasisLayer(2, 10)
        self.dense_layer = DenseLayer(10, 1)

